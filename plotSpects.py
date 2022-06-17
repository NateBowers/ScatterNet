#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

#This returns 3 numbers and 6 lists. Namely in num_list_list, num_list_list, num_list_list

def read_file(filename):
	print('Reading: ' , filename)
	fo = open(filename,'r')
	line1 = fo.readline()
	print("line1: " , line1)
	line2 = fo.readline()
	print(line2)
	line3 = fo.readline()
	print(line3)
	name = '/'.join([str(round(float(x),1)) for x in fo.readline()[:-2].split(',')])
	print(name)
	a1 = [float(x) for x in fo.readline()[:-2].split(',')]
	p1 = [float(x) for x in fo.readline()[:-2].split(',')]
	d1 = [e1 - e2 for (e1,e2) in zip(a1,p1)]
	print(a1[0])
	print(p1[0])
	print(d1[0])
	return name,a1,p1,d1	

def plotFile(filename):
	name,a1,p1,d1 = read_file(filename)
	fig, (ax1, ax2) = plt.subplots(2,gridspec_kw={'height_ratios': [3, 1]})
	fig.suptitle('Comparing spectrums')
	legend1 = [str(name) + "_actual", str(name) + "_predicted"]
	legend2 = ["actual - predicted"]
	ax1.plot(range(400,802,2),a1)
	ax1.plot(range(400,802,2),p1)
	ax2.axhline(y=0,color='black',linewidth=1)
	ax2.plot(range(400,802,2),d1)
	plt.ylabel('Cross Scatting Amplitude')
	plt.xlabel("Wavelength (nm)")
	ax1.legend(legend1)
	ax2.legend(legend2)
	plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--filename",type=str,default='results/8_layer_tio2/test_out_file_20.txt') # 

    args = parser.parse_args()
    dict = vars(args)
    kwargs = {  
            'filename':dict['filename']
            }
    plotFile(**kwargs)




