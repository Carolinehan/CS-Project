import chainer.functions as f
import numpy as np
t=np.array([[[1.0,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
a=f.max(t, axis=-2, keepdims=False)
print a