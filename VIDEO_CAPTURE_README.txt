Conveyor Belief VLA — automatic MP4 capture, v1.4

Apply patch in your existing ~/conveyor-belief-vla directory:
  unzip -o ~/Downloads/conveyor-belief-vla-v1.4-video-patch.zip

Then:
  command -v ffmpeg || (sudo apt update && sudo apt install -y ffmpeg)
  bash scripts/run_demo_video.sh 0.0

Video:
  xdg-open data/latest_demo/demonstration.mp4

Use 0.04 instead of 0.0 for moving conveyor. No Docker rebuild needed.
Read docs/video_capture.md for details.
