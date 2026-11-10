// VGC10 four-cup geometric suction proxy for Gazebo Harmonic (gz-sim8).
// IMPORTANT: not a pressure model or a physical contact sensor.
// Starts DETACHED. Creates a physics fixed joint only after a valid grasp pose
// is held with both vacuum channels enabled. Safe for simulation only.
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <sdf/Element.hh>
#include <cmath>
#include <sstream>
#include <string>

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/transport/Node.hh>

namespace conveyor
{
class Vgc10Vacuum final : public gz::sim::System,
                         public gz::sim::ISystemConfigure,
                         public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity &,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &,
                 gz::sim::EventManager &) override
  {
    this->robotName = _sdf->Get<std::string>("robot_model", "uf850").first;
    this->parentName = _sdf->Get<std::string>("parent_link", "link6").first;
    this->objectName = _sdf->Get<std::string>("object_model", "upper_box").first;
    this->objectLinkName = _sdf->Get<std::string>("object_link", "box_link").first;
    this->pitchX = _sdf->Get<double>("cup_pitch_x", 0.05).first;
    this->pitchY = _sdf->Get<double>("cup_pitch_y", 0.05).first;
    this->boxSizeX = _sdf->Get<double>("box_size_x", 0.09).first;
    this->boxSizeY = _sdf->Get<double>("box_size_y", 0.09).first;
    this->boxSizeZ = _sdf->Get<double>("box_size_z", 0.06).first;
    this->contactTolerance = _sdf->Get<double>("contact_tolerance", 0.007).first;
    this->stableTime = _sdf->Get<double>("stable_contact_s", 0.15).first;
    this->toolOffset = _sdf->Get<gz::math::Vector3d>("tool_offset", gz::math::Vector3d(0, 0, -0.097)).first;
    this->suctionAxis = _sdf->Get<gz::math::Vector3d>("suction_axis", gz::math::Vector3d(0, 0, -1)).first;
    this->node.Subscribe("/vgc10/channel_a/enable", &Vgc10Vacuum::OnA, this);
    this->node.Subscribe("/vgc10/channel_b/enable", &Vgc10Vacuum::OnB, this);
    this->attachedPub = this->node.Advertise<gz::msgs::Boolean>("/vgc10/attached");
    this->contactPub = this->node.Advertise<gz::msgs::StringMsg>("/vgc10/cup_contacts");
    this->qualityPub = this->node.Advertise<gz::msgs::Double>("/vgc10/grasp_quality");
    this->statusPub = this->node.Advertise<gz::msgs::StringMsg>("/vgc10/status");
    this->diagnosticsPub = this->node.Advertise<gz::msgs::StringMsg>("/vgc10/diagnostics");
    gzmsg << "VGC10 vacuum proxy ready: four cups, two channels, initially detached.\n";
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused) return;
    const auto now = std::chrono::duration<double>(_info.simTime).count();
    if (now < this->lastTime) this->validSince = -1;
    this->lastTime = now;
    if (this->parent == gz::sim::kNullEntity ||
        this->object == gz::sim::kNullEntity)
      this->FindEntities(_ecm);

    const bool a = this->channelA.load();
    const bool b = this->channelB.load();
    std::array<bool, 4> contacts{{false, false, false, false}};
    bool directionOK = false;
    double xyError = -1.0;
    double heightGap = -100.0;
    double downDot = -1.0;
    if (this->parent != gz::sim::kNullEntity &&
        this->object != gz::sim::kNullEntity)
    {
      const auto flange = gz::sim::worldPose(this->parent, _ecm);
      const auto box = gz::sim::worldPose(this->object, _ecm);
      // The VGC10 cups face -Z in link6's frame in the generated URDF.
      const auto normal = flange.Rot().RotateVector(this->suctionAxis);
      downDot = normal.Dot(gz::math::Vector3d(0, 0, -1));
      directionOK = downDot > 0.94;
      const auto toolWorld = flange.Pos() + flange.Rot().RotateVector(this->toolOffset);
      const auto toolInBox = box.Rot().Inverse().RotateVector(toolWorld - box.Pos());
      xyError = std::hypot(toolInBox.X(), toolInBox.Y());
      heightGap = toolInBox.Z() - this->boxSizeZ / 2;
      const std::array<gz::math::Vector3d, 4> cupOffsets{{
          {-this->pitchX/2,  this->pitchY/2, 0},
          { this->pitchX/2,  this->pitchY/2, 0},
          {-this->pitchX/2, -this->pitchY/2, 0},
          { this->pitchX/2, -this->pitchY/2, 0}
      }};
      for (std::size_t i = 0; i < cupOffsets.size(); ++i)
      {
        const auto cupWorld = flange.Pos() +
          flange.Rot().RotateVector(this->toolOffset + cupOffsets[i]);
        const auto inBox = box.Rot().Inverse().RotateVector(cupWorld - box.Pos());
        // A geometric seal test: each cup center must be above the top face,
        // and its vertical distance to that face must be within tolerance.
        // The full cup disk is required to stay over the box footprint.
        const double cupRadius = 0.0202; // measured STL sealing-lip radius
        contacts[i] = directionOK &&
          std::abs(inBox.Z() - this->boxSizeZ/2) <= this->contactTolerance &&
          std::abs(inBox.X()) + cupRadius <= this->boxSizeX/2 + 1e-6 &&
          std::abs(inBox.Y()) + cupRadius <= this->boxSizeY/2 + 1e-6;
      }
    }
    int activeContacts = 0;
    for (int i=0; i<4; ++i)
      if (contacts[i] && (i < 2 ? a : b)) ++activeContacts;
    const double quality = static_cast<double>(activeContacts) / 4.0;
    const bool validGrasp = a && b && activeContacts >= 3;

    if (this->attached && (!a || !b))
    {
      if (this->joint != gz::sim::kNullEntity)
        _ecm.RequestRemoveEntity(this->joint);
      this->joint = gz::sim::kNullEntity;
      this->attached = false;
      this->validSince = -1;
      gzmsg << "VGC10 detached upper_box (vacuum off).\n";
    }
    else if (!this->attached && validGrasp)
    {
      if (this->validSince < 0) this->validSince = now;
      if (now - this->validSince >= this->stableTime)
      {
        this->joint = _ecm.CreateEntity();
        _ecm.CreateComponent(this->joint,
          gz::sim::components::DetachableJoint({this->parent, this->object, "fixed"}));
        this->attached = true;
        gzmsg << "VGC10 attached upper_box (cup seal geometry valid).\n";
      }
    }
    else if (!this->attached)
    {
      this->validSince = -1;
    }

    if (now - this->lastPublish >= 0.1 || this->lastPublish < 0)
    {
      this->lastPublish = now;
      gz::msgs::Boolean attachedMsg;
      attachedMsg.set_data(this->attached);
      this->attachedPub.Publish(attachedMsg);
      gz::msgs::Double qualityMsg;
      qualityMsg.set_data(quality);
      this->qualityPub.Publish(qualityMsg);
      gz::msgs::StringMsg cupsMsg;
      cupsMsg.set_data(std::string("a1=") + (contacts[0]?"1":"0") +
           ",a2="+(contacts[1]?"1":"0") +
           ",b1="+(contacts[2]?"1":"0") +
           ",b2="+(contacts[3]?"1":"0"));
      this->contactPub.Publish(cupsMsg);
      gz::msgs::StringMsg statusMsg;
      statusMsg.set_data(this->parent == gz::sim::kNullEntity ? "WAITING_FOR_UF850" :
        (this->object == gz::sim::kNullEntity ? "WAITING_FOR_BOX" :
        (this->attached ? "ATTACHED" :
        (validGrasp ? "SEAL_WAIT" : "DETACHED_NO_VALID_SEAL"))));
      this->statusPub.Publish(statusMsg);
      std::ostringstream details;
      details.precision(4);
      details << std::fixed << "xy_error_m=" << xyError
              << ",height_gap_m=" << heightGap
              << ",down_dot=" << downDot
              << ",active_cups=" << activeContacts
              << ",a=" << (a?1:0) << ",b=" << (b?1:0);
      gz::msgs::StringMsg diagnosticMsg;
      diagnosticMsg.set_data(details.str());
      this->diagnosticsPub.Publish(diagnosticMsg);
    }
  }
private:
  void OnA(const gz::msgs::Boolean &_msg) { this->channelA.store(_msg.data()); }
  void OnB(const gz::msgs::Boolean &_msg) { this->channelB.store(_msg.data()); }
  void FindEntities(const gz::sim::EntityComponentManager &_ecm)
  {
    const auto robot = _ecm.EntityByComponents(
      gz::sim::components::Model(), gz::sim::components::Name(this->robotName));
    const auto obj = _ecm.EntityByComponents(
      gz::sim::components::Model(), gz::sim::components::Name(this->objectName));
    if (robot != gz::sim::kNullEntity)
      this->parent = gz::sim::Model(robot).LinkByName(_ecm, this->parentName);
    if (obj != gz::sim::kNullEntity)
      this->object = gz::sim::Model(obj).LinkByName(_ecm, this->objectLinkName);
  }
  std::string robotName, parentName, objectName, objectLinkName;
  gz::math::Vector3d toolOffset{0, 0, -0.097};
  gz::math::Vector3d suctionAxis{0, 0, -1};
  double pitchX{0.05}, pitchY{0.05};
  double boxSizeX{0.09}, boxSizeY{0.09}, boxSizeZ{0.06};
  double contactTolerance{0.007}, stableTime{0.15};
  double validSince{-1}, lastPublish{-1}, lastTime{0};
  std::atomic<bool> channelA{false}, channelB{false};
  bool attached{false};
  gz::sim::Entity parent{gz::sim::kNullEntity};
  gz::sim::Entity object{gz::sim::kNullEntity};
  gz::sim::Entity joint{gz::sim::kNullEntity};
  gz::transport::Node node;
  gz::transport::Node::Publisher attachedPub, contactPub, qualityPub, statusPub, diagnosticsPub;
};
} // namespace conveyor

GZ_ADD_PLUGIN(conveyor::Vgc10Vacuum, gz::sim::System,
              conveyor::Vgc10Vacuum::ISystemConfigure,
              conveyor::Vgc10Vacuum::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(conveyor::Vgc10Vacuum, "conveyor::Vgc10Vacuum")
