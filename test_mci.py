import ctypes, time, os
p = os.path.abspath('test.mp3')
res_open = ctypes.windll.winmm.mciSendStringW(f'open "{p}" alias tts', None, 0, None)
print("open res:", res_open)
res_set = ctypes.windll.winmm.mciSendStringW('setaudio tts volume to 1000', None, 0, None)
res_play = ctypes.windll.winmm.mciSendStringW('play tts wait', None, 0, None)
print("play res:", res_play)
ctypes.windll.winmm.mciSendStringW('close tts', None, 0, None)
