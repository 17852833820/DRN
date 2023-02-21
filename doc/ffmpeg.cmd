PNG>YUV
ffmpeg -f image2 -i demovideo/stuttgart_00_000000_%6d_leftImg8bit.png -s 2048x1024 -r 30  -pix_fmt yuvj420p  demovideo_2048x1024.yuv
play
ffplay -video_size 2048x1024 -i demovideo_2048x1024.yuv
查看video信息,如帧数
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 -video_size 2048x1024  demovideo_2048x1024.yuv
719
将yuv控制crf编码成mp4
ffmpeg.exe -threads 16 -f rawvideo -framerate 17 -video_size 2048x1024 -pix_fmt yuv420p -i ../test.yuv  -profile:v baseline  -c:v libx264 -crf 30 test-crf=30.mp4
视频 分析工具
http://www.codecian.com/downloads.html
https://blog.csdn.net/eydwyz/article/details/113399702
1.mp4转yuv
# ffmpeg -i guomei.mp4 -s 864x486 -pix_fmt yuv420p guomei.yuv
注意：
-s：设置yuv数据的分辨率
-pix_fmt：设置yuv数据的具体格式
计算psnr
ffmpeg -s 2048x1024 -i ../test.yuv -s 2048x1024 -i  test-crf=5.yuv -lavfi psnr="stats_file=psnr-crf=5.log" -f null - 

ffmpeg -i videoplayback.webm -crf 0 video.mp4
ffmpeg -ss 9:00 -i video.mp4 -t 60 -c:v copy -c:a copy video9-10.mp4
ffmpeg -i video9-10.mp4 -vf scale=2048:1024 video_2048x1024_9-10.mp4 -hide_banner
ffmpeg -i video_2048x1024_9-10.mp4 -r 17 video_2048x1024_9-10_fps17.mp4
ffmpeg -r 17 -i video_2048x1024_9-10_fps17.mp4 -pix_fmt yuvj420p -q:v 2 -vsync 0 -start_number 0 -f image2 -s 2048x1024 ./raw/%5d.png