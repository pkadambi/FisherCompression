'''
This script calculates how much 3-sigma gaussian noise n-bit quantized value can take when the quantization is [-1,1]

'''

q_max = 1
q_min = -1


'''

NOTE: This is an invalid experiment!! The noise is added on TOP of the quantization, it is NOT added before quantization!

Therefore, any noise tolerace would come from quantizing the analog sum (low precision ADC). There might be some CLT stuff
here but it's hard to say... 


'''
diff = q_max-q_min

noise_sigma = []
for n_bits in range(1,9):

    level_spacing = diff/(2 ** n_bits)
    # sigma_target = level_spacing/6

    sigma_target = 1/(2 ** (n_bits-1))
    noise_sigma.append(sigma_target)

import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.arange(8)+1, np.fliplr(np.expand_dims(noise_sigma, axis=1)))
plt.title('Inherent Noise Robustness: \nNumber of Bits vs Variance of Noise Tolerated')
plt.grid(True)
plt.xlabel('Number of Bits (Weight)')
plt.ylabel('Tolerated Noise Variance/Variance of Weight')
plt.show()




