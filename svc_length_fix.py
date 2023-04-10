import ffmpeg

# 用于规避hifi-gan-bwe切片bug，在音频末尾添加30s静音满足切片最短长度
input_file = 'svc_out.wav'
output_file = 'svc_out_fix.wav'
duration = 30

audio = ffmpeg.input(input_file)
silence = ffmpeg.input('anullsrc', f='lavfi', t=duration)
output = ffmpeg.concat(audio, silence, v=0, a=1).output(output_file)

ffmpeg.run(output)