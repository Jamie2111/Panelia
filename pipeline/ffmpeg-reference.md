# FFmpeg Reference

Encode the motion-comic camera pass from raw RGB frames:

```bash
ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 1080x1920 -r 24 -i - \
  -an -c:v libx264 -pix_fmt yuv420p camera_pass.mp4
```

Mux narration onto the rendered camera pass:

```bash
ffmpeg -y -i camera_pass.mp4 -i narration.wav \
  -c:v copy -c:a aac -shortest final.mp4
```

Mix background music under narration:

```bash
ffmpeg -y -i final.mp4 -stream_loop -1 -i music.mp3 \
  -filter_complex "[1:a]volume=0.2,afade=t=in:st=0:d=1,afade=t=out:st=29:d=2[music];[0:a][music]amix=inputs=2:duration=first[aout]" \
  -map 0:v -map "[aout]" -c:v copy -c:a aac final_with_music.mp4
```
