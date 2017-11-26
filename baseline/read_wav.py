import sys
import wave
import numpy
import struct
from scipy.interpolate import interp1d

if len(sys.argv) < 2:
    print("Usage: read_wav.py wavfile")
    exit(1)
else:
    wavfile = sys.argv[1]

speed = float(0.8)
f = wave.open(wavfile, "rb")
params = f.getparams()
[nchannels, sampwidth, framerate, nframes] = params[:4]

str_data = f.readframes(nframes)
f.close()

wave_data = numpy.fromstring(str_data, dtype=numpy.short)
wave_data.shape = [-1, 2]
wave_data = wave_data.T
# time = numpy.arange(0, nframes) * (1.0 / framerate)
# len_time = len(time) / 2
# time = time[0:len_time]

frames = len(wave_data[0])
# for i in range(0, frames):
# 	print wave_data[0][i]
# 	print wave_data[1][i]
newFrames = int(frames * speed)
newData = numpy.zeros(2 * newFrames).reshape(2, newFrames)

data = []
print(newFrames, frames)
for i in range(0, 2):
    time_axis = numpy.arange(frames)
    interp = interp1d(time_axis, wave_data[i], kind='linear', bounds_error=False, fill_value=0)
    time_axis = numpy.arange(newFrames) * (frames / newFrames)
    newData[i] = newData[i] + interp(time_axis)

for j in range(0, len(newData[0])):
    temp = struct.pack('h', int(newData[0][j]))
    data.append(temp)
    temp = struct.pack('h', int(newData[1][j]))
    data.append(temp)
data_string = b"".join(data)
# print data_string
wavfile = wavfile.strip(".overall") + "_dur.overall"
# print wavfile
fout = wave.open(wavfile, "wb")
fout.setparams([nchannels, sampwidth, framerate, 0, "NONE", "not compressed"])
fout.writeframes(data_string)
fout.close()
