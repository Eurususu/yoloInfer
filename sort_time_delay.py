data = [
{ "name" : "Reformatting CopyNode for Input Tensor 0 to /model.0/conv/Conv", "timeMs" : 0.029696, "averageMs" : 0.029696, "percentage" : 0.828453 }
, { "name" : "/model.0/conv/Conv", "timeMs" : 0.03584, "averageMs" : 0.03584, "percentage" : 0.999857 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul)", "timeMs" : 0.044032, "averageMs" : 0.044032, "percentage" : 1.2284 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.1/conv/Conv + PWN(PWN(/model.1/act/Sigmoid), /model.1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.1/conv/Conv + PWN(PWN(/model.1/act/Sigmoid), /model.1/act/Mul)", "timeMs" : 0.057344, "averageMs" : 0.057344, "percentage" : 1.59977 }
, { "name" : "/model.2/cv1/conv/Conv + PWN(PWN(/model.2/cv1/act/Sigmoid), /model.2/cv1/act/Mul)", "timeMs" : 0.029696, "averageMs" : 0.029696, "percentage" : 0.828453 }
, { "name" : "/model.2/Slice", "timeMs" : 0.014336, "averageMs" : 0.014336, "percentage" : 0.399943 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.2/m.0/cv1/conv/Conv + PWN(PWN(/model.2/m.0/cv1/act/Sigmoid), /model.2/m.0/cv1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.2/m.0/cv1/conv/Conv + PWN(PWN(/model.2/m.0/cv1/act/Sigmoid), /model.2/m.0/cv1/act/Mul)", "timeMs" : 0.037888, "averageMs" : 0.037888, "percentage" : 1.05699 }
, { "name" : "Reformatting CopyNode for Input Tensor 1 to /model.2/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.2/m.0/cv2/act/Sigmoid), /model.2/m.0/cv2/act/Mul), /model.2/m.0/Add)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.2/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.2/m.0/cv2/act/Sigmoid), /model.2/m.0/cv2/act/Mul), /model.2/m.0/Add)", "timeMs" : 0.038912, "averageMs" : 0.038912, "percentage" : 1.08556 }
, { "name" : "/model.2/cv1/act/Mul_output_0 copy", "timeMs" : 0.025664, "averageMs" : 0.025664, "percentage" : 0.715969 }
, { "name" : "/model.2/m.0/Add_output_0 copy", "timeMs" : 0.023488, "averageMs" : 0.023488, "percentage" : 0.655263 }
, { "name" : "/model.2/cv2/conv/Conv + PWN(PWN(/model.2/cv2/act/Sigmoid), /model.2/cv2/act/Mul)", "timeMs" : 0.034816, "averageMs" : 0.034816, "percentage" : 0.97129 }
, { "name" : "/model.3/conv/Conv + PWN(PWN(/model.3/act/Sigmoid), /model.3/act/Mul)", "timeMs" : 0.043008, "averageMs" : 0.043008, "percentage" : 1.19983 }
, { "name" : "/model.4/cv1/conv/Conv + PWN(PWN(/model.4/cv1/act/Sigmoid), /model.4/cv1/act/Mul)", "timeMs" : 0.018432, "averageMs" : 0.018432, "percentage" : 0.514212 }
, { "name" : "Reformatting CopyNode for Output Tensor 0 to /model.4/cv1/conv/Conv + PWN(PWN(/model.4/cv1/act/Sigmoid), /model.4/cv1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.4/Slice", "timeMs" : 0.013312, "averageMs" : 0.013312, "percentage" : 0.371376 }
, { "name" : "/model.4/m.0/cv1/conv/Conv + PWN(PWN(/model.4/m.0/cv1/act/Sigmoid), /model.4/m.0/cv1/act/Mul)", "timeMs" : 0.022528, "averageMs" : 0.022528, "percentage" : 0.628482 }
, { "name" : "/model.4/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.4/m.0/cv2/act/Sigmoid), /model.4/m.0/cv2/act/Mul), /model.4/m.0/Add)", "timeMs" : 0.023552, "averageMs" : 0.023552, "percentage" : 0.657049 }
, { "name" : "Reformatting CopyNode for Output Tensor 0 to /model.4/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.4/m.0/cv2/act/Sigmoid), /model.4/m.0/cv2/act/Mul), /model.4/m.0/Add)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.4/m.1/cv1/conv/Conv + PWN(PWN(/model.4/m.1/cv1/act/Sigmoid), /model.4/m.1/cv1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.4/m.1/cv1/conv/Conv + PWN(PWN(/model.4/m.1/cv1/act/Sigmoid), /model.4/m.1/cv1/act/Mul)", "timeMs" : 0.022528, "averageMs" : 0.022528, "percentage" : 0.628482 }
, { "name" : "Reformatting CopyNode for Input Tensor 1 to /model.4/m.1/cv2/conv/Conv + PWN(PWN(PWN(/model.4/m.1/cv2/act/Sigmoid), /model.4/m.1/cv2/act/Mul), /model.4/m.1/Add)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.4/m.1/cv2/conv/Conv + PWN(PWN(PWN(/model.4/m.1/cv2/act/Sigmoid), /model.4/m.1/cv2/act/Mul), /model.4/m.1/Add)", "timeMs" : 0.021504, "averageMs" : 0.021504, "percentage" : 0.599914 }
, { "name" : "/model.4/cv1/act/Mul_output_0 copy", "timeMs" : 0.015648, "averageMs" : 0.015648, "percentage" : 0.436545 }
, { "name" : "/model.4/m.0/Add_output_0 copy", "timeMs" : 0.009952, "averageMs" : 0.009952, "percentage" : 0.277639 }
, { "name" : "/model.4/m.1/Add_output_0 copy", "timeMs" : 0.014336, "averageMs" : 0.014336, "percentage" : 0.399943 }
, { "name" : "/model.4/cv2/conv/Conv + PWN(PWN(/model.4/cv2/act/Sigmoid), /model.4/cv2/act/Mul)", "timeMs" : 0.026624, "averageMs" : 0.026624, "percentage" : 0.742751 }
, { "name" : "/model.5/conv/Conv + PWN(PWN(/model.5/act/Sigmoid), /model.5/act/Mul)", "timeMs" : 0.032768, "averageMs" : 0.032768, "percentage" : 0.914155 }
, { "name" : "/model.6/cv1/conv/Conv + PWN(PWN(/model.6/cv1/act/Sigmoid), /model.6/cv1/act/Mul)", "timeMs" : 0.01536, "averageMs" : 0.01536, "percentage" : 0.42851 }
, { "name" : "Reformatting CopyNode for Output Tensor 0 to /model.6/cv1/conv/Conv + PWN(PWN(/model.6/cv1/act/Sigmoid), /model.6/cv1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.6/Slice", "timeMs" : 0.009216, "averageMs" : 0.009216, "percentage" : 0.257106 }
, { "name" : "/model.6/m.0/cv1/conv/Conv + PWN(PWN(/model.6/m.0/cv1/act/Sigmoid), /model.6/m.0/cv1/act/Mul)", "timeMs" : 0.02048, "averageMs" : 0.02048, "percentage" : 0.571347 }
, { "name" : "/model.6/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.6/m.0/cv2/act/Sigmoid), /model.6/m.0/cv2/act/Mul), /model.6/m.0/Add)", "timeMs" : 0.02048, "averageMs" : 0.02048, "percentage" : 0.571347 }
, { "name" : "/model.6/m.1/cv1/conv/Conv + PWN(PWN(/model.6/m.1/cv1/act/Sigmoid), /model.6/m.1/cv1/act/Mul)", "timeMs" : 0.02048, "averageMs" : 0.02048, "percentage" : 0.571347 }
, { "name" : "/model.6/m.1/cv2/conv/Conv + PWN(PWN(PWN(/model.6/m.1/cv2/act/Sigmoid), /model.6/m.1/cv2/act/Mul), /model.6/m.1/Add)", "timeMs" : 0.02048, "averageMs" : 0.02048, "percentage" : 0.571347 }
, { "name" : "/model.6/cv1/act/Mul_output_0 copy", "timeMs" : 0.013312, "averageMs" : 0.013312, "percentage" : 0.371376 }
, { "name" : "/model.6/m.0/Add_output_0 copy", "timeMs" : 0.009216, "averageMs" : 0.009216, "percentage" : 0.257106 }
, { "name" : "/model.6/m.1/Add_output_0 copy", "timeMs" : 0.008192, "averageMs" : 0.008192, "percentage" : 0.228539 }
, { "name" : "/model.6/cv2/conv/Conv + PWN(PWN(/model.6/cv2/act/Sigmoid), /model.6/cv2/act/Mul)", "timeMs" : 0.022528, "averageMs" : 0.022528, "percentage" : 0.628482 }
, { "name" : "/model.7/conv/Conv + PWN(PWN(/model.7/act/Sigmoid), /model.7/act/Mul)", "timeMs" : 0.032768, "averageMs" : 0.032768, "percentage" : 0.914155 }
, { "name" : "/model.8/cv1/conv/Conv + PWN(PWN(/model.8/cv1/act/Sigmoid), /model.8/cv1/act/Mul)", "timeMs" : 0.014336, "averageMs" : 0.014336, "percentage" : 0.399943 }
, { "name" : "Reformatting CopyNode for Output Tensor 0 to /model.8/cv1/conv/Conv + PWN(PWN(/model.8/cv1/act/Sigmoid), /model.8/cv1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.8/m.0/cv1/conv/Conv + PWN(PWN(/model.8/m.0/cv1/act/Sigmoid), /model.8/m.0/cv1/act/Mul)", "timeMs" : 0.007168, "averageMs" : 0.007168, "percentage" : 0.199971 }
, { "name" : "/model.8/m.0/cv1/conv/Conv + PWN(PWN(/model.8/m.0/cv1/act/Sigmoid), /model.8/m.0/cv1/act/Mul)", "timeMs" : 0.022528, "averageMs" : 0.022528, "percentage" : 0.628482 }
, { "name" : "/model.8/m.0/cv2/conv/Conv", "timeMs" : 0.019456, "averageMs" : 0.019456, "percentage" : 0.54278 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to PWN(PWN(PWN(/model.8/m.0/cv2/act/Sigmoid), /model.8/m.0/cv2/act/Mul), /model.8/m.0/Add)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "PWN(PWN(PWN(/model.8/m.0/cv2/act/Sigmoid), /model.8/m.0/cv2/act/Mul), /model.8/m.0/Add)", "timeMs" : 0.00512, "averageMs" : 0.00512, "percentage" : 0.142837 }
, { "name" : "/model.8/cv1/act/Mul_output_0 copy", "timeMs" : 0.009216, "averageMs" : 0.009216, "percentage" : 0.257106 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.8/cv2/conv/Conv + PWN(PWN(/model.8/cv2/act/Sigmoid), /model.8/cv2/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.8/cv2/conv/Conv + PWN(PWN(/model.8/cv2/act/Sigmoid), /model.8/cv2/act/Mul)", "timeMs" : 0.017408, "averageMs" : 0.017408, "percentage" : 0.485645 }
, { "name" : "/model.9/cv1/conv/Conv + PWN(PWN(/model.9/cv1/act/Sigmoid), /model.9/cv1/act/Mul)", "timeMs" : 0.01024, "averageMs" : 0.01024, "percentage" : 0.285673 }
, { "name" : "Reformatting CopyNode for Output Tensor 0 to /model.9/cv1/conv/Conv + PWN(PWN(/model.9/cv1/act/Sigmoid), /model.9/cv1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.9/m/MaxPool", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.9/m/MaxPool", "timeMs" : 0.008192, "averageMs" : 0.008192, "percentage" : 0.228539 }
, { "name" : "/model.9/m_1/MaxPool", "timeMs" : 0.00736, "averageMs" : 0.00736, "percentage" : 0.205328 }
, { "name" : "/model.9/m_2/MaxPool", "timeMs" : 0.008, "averageMs" : 0.008, "percentage" : 0.223182 }
, { "name" : "/model.9/cv1/act/Mul_output_0 copy", "timeMs" : 0.006144, "averageMs" : 0.006144, "percentage" : 0.171404 }
, { "name" : "/model.9/m/MaxPool_output_0 copy", "timeMs" : 0.006144, "averageMs" : 0.006144, "percentage" : 0.171404 }
, { "name" : "/model.9/m_1/MaxPool_output_0 copy", "timeMs" : 0.006144, "averageMs" : 0.006144, "percentage" : 0.171404 }
, { "name" : "/model.9/cv2/conv/Conv + PWN(PWN(/model.9/cv2/act/Sigmoid), /model.9/cv2/act/Mul)", "timeMs" : 0.019648, "averageMs" : 0.019648, "percentage" : 0.548136 }
, { "name" : "Reformatting CopyNode for Output Tensor 0 to /model.9/cv2/conv/Conv + PWN(PWN(/model.9/cv2/act/Sigmoid), /model.9/cv2/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.10/Resize", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.10/Resize", "timeMs" : 0.007168, "averageMs" : 0.007168, "percentage" : 0.199971 }
, { "name" : "/model.10/Resize_output_0 copy", "timeMs" : 0.023552, "averageMs" : 0.023552, "percentage" : 0.657049 }
, { "name" : "/model.12/cv1/conv/Conv + PWN(PWN(/model.12/cv1/act/Sigmoid), /model.12/cv1/act/Mul)", "timeMs" : 0.026624, "averageMs" : 0.026624, "percentage" : 0.742751 }
, { "name" : "/model.12/m.0/cv1/conv/Conv + PWN(PWN(/model.12/m.0/cv1/act/Sigmoid), /model.12/m.0/cv1/act/Mul)", "timeMs" : 0.021504, "averageMs" : 0.021504, "percentage" : 0.599914 }
, { "name" : "/model.12/m.0/cv2/conv/Conv + PWN(PWN(/model.12/m.0/cv2/act/Sigmoid), /model.12/m.0/cv2/act/Mul)", "timeMs" : 0.019456, "averageMs" : 0.019456, "percentage" : 0.54278 }
, { "name" : "/model.12/cv1/act/Mul_output_0 copy", "timeMs" : 0.013312, "averageMs" : 0.013312, "percentage" : 0.371376 }
, { "name" : "/model.12/cv2/conv/Conv + PWN(PWN(/model.12/cv2/act/Sigmoid), /model.12/cv2/act/Mul)", "timeMs" : 0.018432, "averageMs" : 0.018432, "percentage" : 0.514212 }
, { "name" : "Reformatting CopyNode for Output Tensor 0 to /model.12/cv2/conv/Conv + PWN(PWN(/model.12/cv2/act/Sigmoid), /model.12/cv2/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.13/Resize", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.13/Resize", "timeMs" : 0.016384, "averageMs" : 0.016384, "percentage" : 0.457078 }
, { "name" : "/model.13/Resize_output_0 copy", "timeMs" : 0.027648, "averageMs" : 0.027648, "percentage" : 0.771318 }
, { "name" : "/model.15/cv1/conv/Conv + PWN(PWN(/model.15/cv1/act/Sigmoid), /model.15/cv1/act/Mul)", "timeMs" : 0.029696, "averageMs" : 0.029696, "percentage" : 0.828453 }
, { "name" : "/model.15/m.0/cv1/conv/Conv + PWN(PWN(/model.15/m.0/cv1/act/Sigmoid), /model.15/m.0/cv1/act/Mul)", "timeMs" : 0.023552, "averageMs" : 0.023552, "percentage" : 0.657049 }
, { "name" : "/model.15/m.0/cv2/conv/Conv + PWN(PWN(/model.15/m.0/cv2/act/Sigmoid), /model.15/m.0/cv2/act/Mul)", "timeMs" : 0.021504, "averageMs" : 0.021504, "percentage" : 0.599914 }
, { "name" : "/model.15/cv1/act/Mul_output_0 copy", "timeMs" : 0.022528, "averageMs" : 0.022528, "percentage" : 0.628482 }
, { "name" : "/model.15/cv2/conv/Conv + PWN(PWN(/model.15/cv2/act/Sigmoid), /model.15/cv2/act/Mul)", "timeMs" : 0.021504, "averageMs" : 0.021504, "percentage" : 0.599914 }
, { "name" : "Reformatting CopyNode for Output Tensor 0 to /model.15/cv2/conv/Conv + PWN(PWN(/model.15/cv2/act/Sigmoid), /model.15/cv2/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.16/Resize", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.16/Resize", "timeMs" : 0.026624, "averageMs" : 0.026624, "percentage" : 0.742751 }
, { "name" : "/model.16/Resize_output_0 copy", "timeMs" : 0.047104, "averageMs" : 0.047104, "percentage" : 1.3141 }
, { "name" : "/model.18/cv1/conv/Conv + PWN(PWN(/model.18/cv1/act/Sigmoid), /model.18/cv1/act/Mul)", "timeMs" : 0.050176, "averageMs" : 0.050176, "percentage" : 1.3998 }
, { "name" : "/model.18/m.0/cv1/conv/Conv + PWN(PWN(/model.18/m.0/cv1/act/Sigmoid), /model.18/m.0/cv1/act/Mul)", "timeMs" : 0.036864, "averageMs" : 0.036864, "percentage" : 1.02842 }
, { "name" : "/model.18/m.0/cv2/conv/Conv + PWN(PWN(/model.18/m.0/cv2/act/Sigmoid), /model.18/m.0/cv2/act/Mul)", "timeMs" : 0.038912, "averageMs" : 0.038912, "percentage" : 1.08556 }
, { "name" : "/model.18/cv1/act/Mul_output_0 copy", "timeMs" : 0.024576, "averageMs" : 0.024576, "percentage" : 0.685616 }
, { "name" : "/model.18/cv2/conv/Conv + PWN(PWN(/model.18/cv2/act/Sigmoid), /model.18/cv2/act/Mul)", "timeMs" : 0.032768, "averageMs" : 0.032768, "percentage" : 0.914155 }
, { "name" : "/model.19/conv/Conv + PWN(PWN(/model.19/act/Sigmoid), /model.19/act/Mul)", "timeMs" : 0.03072, "averageMs" : 0.03072, "percentage" : 0.85702 }
, { "name" : "/model.15/cv2/act/Mul_output_0 copy", "timeMs" : 0.023552, "averageMs" : 0.023552, "percentage" : 0.657049 }
, { "name" : "/model.21/cv1/conv/Conv + PWN(PWN(/model.21/cv1/act/Sigmoid), /model.21/cv1/act/Mul)", "timeMs" : 0.022528, "averageMs" : 0.022528, "percentage" : 0.628482 }
, { "name" : "/model.21/m.0/cv1/conv/Conv + PWN(PWN(/model.21/m.0/cv1/act/Sigmoid), /model.21/m.0/cv1/act/Mul)", "timeMs" : 0.022528, "averageMs" : 0.022528, "percentage" : 0.628482 }
, { "name" : "/model.21/m.0/cv2/conv/Conv + PWN(PWN(/model.21/m.0/cv2/act/Sigmoid), /model.21/m.0/cv2/act/Mul)", "timeMs" : 0.021504, "averageMs" : 0.021504, "percentage" : 0.599914 }
, { "name" : "/model.21/cv1/act/Mul_output_0 copy", "timeMs" : 0.023552, "averageMs" : 0.023552, "percentage" : 0.657049 }
, { "name" : "/model.21/cv2/conv/Conv + PWN(PWN(/model.21/cv2/act/Sigmoid), /model.21/cv2/act/Mul)", "timeMs" : 0.02048, "averageMs" : 0.02048, "percentage" : 0.571347 }
, { "name" : "/model.22/conv/Conv + PWN(PWN(/model.22/act/Sigmoid), /model.22/act/Mul)", "timeMs" : 0.02048, "averageMs" : 0.02048, "percentage" : 0.571347 }
, { "name" : "/model.12/cv2/act/Mul_output_0 copy", "timeMs" : 0.01344, "averageMs" : 0.01344, "percentage" : 0.374946 }
, { "name" : "/model.24/cv1/conv/Conv + PWN(PWN(/model.24/cv1/act/Sigmoid), /model.24/cv1/act/Mul)", "timeMs" : 0.018304, "averageMs" : 0.018304, "percentage" : 0.510641 }
, { "name" : "/model.24/m.0/cv1/conv/Conv + PWN(PWN(/model.24/m.0/cv1/act/Sigmoid), /model.24/m.0/cv1/act/Mul)", "timeMs" : 0.019456, "averageMs" : 0.019456, "percentage" : 0.54278 }
, { "name" : "/model.24/m.0/cv2/conv/Conv + PWN(PWN(/model.24/m.0/cv2/act/Sigmoid), /model.24/m.0/cv2/act/Mul)", "timeMs" : 0.019456, "averageMs" : 0.019456, "percentage" : 0.54278 }
, { "name" : "/model.24/cv1/act/Mul_output_0 copy", "timeMs" : 0.014336, "averageMs" : 0.014336, "percentage" : 0.399943 }
, { "name" : "/model.24/cv2/conv/Conv + PWN(PWN(/model.24/cv2/act/Sigmoid), /model.24/cv2/act/Mul)", "timeMs" : 0.016384, "averageMs" : 0.016384, "percentage" : 0.457078 }
, { "name" : "/model.25/conv/Conv + PWN(PWN(/model.25/act/Sigmoid), /model.25/act/Mul)", "timeMs" : 0.024576, "averageMs" : 0.024576, "percentage" : 0.685616 }
, { "name" : "/model.9/cv2/act/Mul_output_0 copy", "timeMs" : 0.009216, "averageMs" : 0.009216, "percentage" : 0.257106 }
, { "name" : "/model.27/cv1/conv/Conv + PWN(PWN(/model.27/cv1/act/Sigmoid), /model.27/cv1/act/Mul)", "timeMs" : 0.016384, "averageMs" : 0.016384, "percentage" : 0.457078 }
, { "name" : "/model.27/m.0/cv1/conv/Conv + PWN(PWN(/model.27/m.0/cv1/act/Sigmoid), /model.27/m.0/cv1/act/Mul)", "timeMs" : 0.022528, "averageMs" : 0.022528, "percentage" : 0.628482 }
, { "name" : "/model.27/m.0/cv2/conv/Conv + PWN(PWN(/model.27/m.0/cv2/act/Sigmoid), /model.27/m.0/cv2/act/Mul)", "timeMs" : 0.022528, "averageMs" : 0.022528, "percentage" : 0.628482 }
, { "name" : "/model.27/cv1/act/Mul_output_0 copy", "timeMs" : 0.009216, "averageMs" : 0.009216, "percentage" : 0.257106 }
, { "name" : "/model.27/cv2/conv/Conv + PWN(PWN(/model.27/cv2/act/Sigmoid), /model.27/cv2/act/Mul)", "timeMs" : 0.01536, "averageMs" : 0.01536, "percentage" : 0.42851 }
, { "name" : "/model.28/cv2.3/cv2.3.0/conv/Conv || /model.28/cv3.3/cv3.3.0/conv/Conv", "timeMs" : 0.029696, "averageMs" : 0.029696, "percentage" : 0.828453 }
, { "name" : "Reformatting CopyNode for Output Tensor 0 to /model.28/cv2.3/cv2.3.0/conv/Conv || /model.28/cv3.3/cv3.3.0/conv/Conv", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "PWN(PWN(/model.28/cv2.3/cv2.3.0/act/Sigmoid), /model.28/cv2.3/cv2.3.0/act/Mul)", "timeMs" : 0.00336, "averageMs" : 0.00336, "percentage" : 0.0937366 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.28/cv2.3/cv2.3.1/conv/Conv + PWN(PWN(/model.28/cv2.3/cv2.3.1/act/Sigmoid), /model.28/cv2.3/cv2.3.1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/cv2.3/cv2.3.1/conv/Conv + PWN(PWN(/model.28/cv2.3/cv2.3.1/act/Sigmoid), /model.28/cv2.3/cv2.3.1/act/Mul)", "timeMs" : 0.009216, "averageMs" : 0.009216, "percentage" : 0.257106 }
, { "name" : "/model.28/cv2.3/cv2.3.2/Conv", "timeMs" : 0.006144, "averageMs" : 0.006144, "percentage" : 0.171404 }
, { "name" : "PWN(PWN(/model.28/cv3.3/cv3.3.0/act/Sigmoid), /model.28/cv3.3/cv3.3.0/act/Mul)", "timeMs" : 0.004096, "averageMs" : 0.004096, "percentage" : 0.114269 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.28/cv3.3/cv3.3.1/conv/Conv + PWN(PWN(/model.28/cv3.3/cv3.3.1/act/Sigmoid), /model.28/cv3.3/cv3.3.1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/cv3.3/cv3.3.1/conv/Conv + PWN(PWN(/model.28/cv3.3/cv3.3.1/act/Sigmoid), /model.28/cv3.3/cv3.3.1/act/Mul)", "timeMs" : 0.008192, "averageMs" : 0.008192, "percentage" : 0.228539 }
, { "name" : "/model.28/cv3.3/cv3.3.2/Conv", "timeMs" : 0.006144, "averageMs" : 0.006144, "percentage" : 0.171404 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.28/Reshape_3", "timeMs" : 0.004096, "averageMs" : 0.004096, "percentage" : 0.114269 }
, { "name" : "/model.28/Reshape_3", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/Reshape_3_copy_output", "timeMs" : 0.272384, "averageMs" : 0.272384, "percentage" : 7.59891 }
, { "name" : "/model.28/cv2.2/cv2.2.0/conv/Conv || /model.28/cv3.2/cv3.2.0/conv/Conv", "timeMs" : 0.031744, "averageMs" : 0.031744, "percentage" : 0.885588 }
, { "name" : "Reformatting CopyNode for Output Tensor 0 to /model.28/cv2.2/cv2.2.0/conv/Conv || /model.28/cv3.2/cv3.2.0/conv/Conv", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "PWN(PWN(/model.28/cv2.2/cv2.2.0/act/Sigmoid), /model.28/cv2.2/cv2.2.0/act/Mul)", "timeMs" : 0.004096, "averageMs" : 0.004096, "percentage" : 0.114269 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.28/cv2.2/cv2.2.1/conv/Conv + PWN(PWN(/model.28/cv2.2/cv2.2.1/act/Sigmoid), /model.28/cv2.2/cv2.2.1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/cv2.2/cv2.2.1/conv/Conv + PWN(PWN(/model.28/cv2.2/cv2.2.1/act/Sigmoid), /model.28/cv2.2/cv2.2.1/act/Mul)", "timeMs" : 0.012288, "averageMs" : 0.012288, "percentage" : 0.342808 }
, { "name" : "/model.28/cv2.2/cv2.2.2/Conv", "timeMs" : 0.007168, "averageMs" : 0.007168, "percentage" : 0.199971 }
, { "name" : "PWN(PWN(/model.28/cv3.2/cv3.2.0/act/Sigmoid), /model.28/cv3.2/cv3.2.0/act/Mul)", "timeMs" : 0.00512, "averageMs" : 0.00512, "percentage" : 0.142837 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.28/cv3.2/cv3.2.1/conv/Conv + PWN(PWN(/model.28/cv3.2/cv3.2.1/act/Sigmoid), /model.28/cv3.2/cv3.2.1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/cv3.2/cv3.2.1/conv/Conv + PWN(PWN(/model.28/cv3.2/cv3.2.1/act/Sigmoid), /model.28/cv3.2/cv3.2.1/act/Mul)", "timeMs" : 0.012288, "averageMs" : 0.012288, "percentage" : 0.342808 }
, { "name" : "/model.28/cv3.2/cv3.2.2/Conv", "timeMs" : 0.006144, "averageMs" : 0.006144, "percentage" : 0.171404 }
, { "name" : "/model.28/Reshape_2", "timeMs" : 0.008192, "averageMs" : 0.008192, "percentage" : 0.228539 }
, { "name" : "/model.28/cv2.1/cv2.1.0/conv/Conv || /model.28/cv3.1/cv3.1.0/conv/Conv", "timeMs" : 0.057344, "averageMs" : 0.057344, "percentage" : 1.59977 }
, { "name" : "PWN(PWN(/model.28/cv2.1/cv2.1.0/act/Sigmoid), /model.28/cv2.1/cv2.1.0/act/Mul)", "timeMs" : 0.009216, "averageMs" : 0.009216, "percentage" : 0.257106 }
, { "name" : "/model.28/cv2.1/cv2.1.1/conv/Conv + PWN(PWN(/model.28/cv2.1/cv2.1.1/act/Sigmoid), /model.28/cv2.1/cv2.1.1/act/Mul)", "timeMs" : 0.022528, "averageMs" : 0.022528, "percentage" : 0.628482 }
, { "name" : "/model.28/cv2.1/cv2.1.2/Conv", "timeMs" : 0.01024, "averageMs" : 0.01024, "percentage" : 0.285673 }
, { "name" : "PWN(PWN(/model.28/cv3.1/cv3.1.0/act/Sigmoid), /model.28/cv3.1/cv3.1.0/act/Mul)", "timeMs" : 0.009216, "averageMs" : 0.009216, "percentage" : 0.257106 }
, { "name" : "/model.28/cv3.1/cv3.1.1/conv/Conv + PWN(PWN(/model.28/cv3.1/cv3.1.1/act/Sigmoid), /model.28/cv3.1/cv3.1.1/act/Mul)", "timeMs" : 0.022528, "averageMs" : 0.022528, "percentage" : 0.628482 }
, { "name" : "/model.28/cv3.1/cv3.1.2/Conv", "timeMs" : 0.008192, "averageMs" : 0.008192, "percentage" : 0.228539 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.28/Reshape_1", "timeMs" : 0.01024, "averageMs" : 0.01024, "percentage" : 0.285673 }
, { "name" : "/model.28/Reshape_1", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/Reshape_1_copy_output", "timeMs" : 0.009216, "averageMs" : 0.009216, "percentage" : 0.257106 }
, { "name" : "/model.28/cv2.0/cv2.0.0/conv/Conv || /model.28/cv3.0/cv3.0.0/conv/Conv", "timeMs" : 0.115712, "averageMs" : 0.115712, "percentage" : 3.22811 }
, { "name" : "Reformatting CopyNode for Output Tensor 0 to /model.28/cv2.0/cv2.0.0/conv/Conv || /model.28/cv3.0/cv3.0.0/conv/Conv", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "PWN(PWN(/model.28/cv2.0/cv2.0.0/act/Sigmoid), /model.28/cv2.0/cv2.0.0/act/Mul)", "timeMs" : 0.024576, "averageMs" : 0.024576, "percentage" : 0.685616 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.28/cv2.0/cv2.0.1/conv/Conv + PWN(PWN(/model.28/cv2.0/cv2.0.1/act/Sigmoid), /model.28/cv2.0/cv2.0.1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/cv2.0/cv2.0.1/conv/Conv + PWN(PWN(/model.28/cv2.0/cv2.0.1/act/Sigmoid), /model.28/cv2.0/cv2.0.1/act/Mul)", "timeMs" : 0.079872, "averageMs" : 0.079872, "percentage" : 2.22825 }
, { "name" : "/model.28/cv2.0/cv2.0.2/Conv", "timeMs" : 0.028672, "averageMs" : 0.028672, "percentage" : 0.799886 }
, { "name" : "PWN(PWN(/model.28/cv3.0/cv3.0.0/act/Sigmoid), /model.28/cv3.0/cv3.0.0/act/Mul)", "timeMs" : 0.024576, "averageMs" : 0.024576, "percentage" : 0.685616 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.28/cv3.0/cv3.0.1/conv/Conv + PWN(PWN(/model.28/cv3.0/cv3.0.1/act/Sigmoid), /model.28/cv3.0/cv3.0.1/act/Mul)", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/cv3.0/cv3.0.1/conv/Conv + PWN(PWN(/model.28/cv3.0/cv3.0.1/act/Sigmoid), /model.28/cv3.0/cv3.0.1/act/Mul)", "timeMs" : 0.080896, "averageMs" : 0.080896, "percentage" : 2.25682 }
, { "name" : "/model.28/cv3.0/cv3.0.2/Conv", "timeMs" : 0.021504, "averageMs" : 0.021504, "percentage" : 0.599914 }
, { "name" : "Reformatting CopyNode for Input Tensor 0 to /model.28/Reshape", "timeMs" : 0.028672, "averageMs" : 0.028672, "percentage" : 0.799886 }
, { "name" : "/model.28/Reshape", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/Reshape_copy_output", "timeMs" : 0.021504, "averageMs" : 0.021504, "percentage" : 0.599914 }
, { "name" : "/model.28/Reshape_2_output_0 copy", "timeMs" : 0.00512, "averageMs" : 0.00512, "percentage" : 0.142837 }
, { "name" : "/model.28/Reshape_4 + /model.28/Transpose", "timeMs" : 0.048128, "averageMs" : 0.048128, "percentage" : 1.34267 }
, { "name" : "(Unnamed Layer* 592) [Shuffle]", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/Softmax", "timeMs" : 0.305152, "averageMs" : 0.305152, "percentage" : 8.51307 }
, { "name" : "(Unnamed Layer* 594) [Shuffle] + reshape_before_/model.28/MatMul", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/MatMul", "timeMs" : 0.02048, "averageMs" : 0.02048, "percentage" : 0.571347 }
, { "name" : "reshape_after_/model.28/MatMul + (Unnamed Layer* 603) [Shuffle]", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "PWN(/model.28/Neg)", "timeMs" : 0.004096, "averageMs" : 0.004096, "percentage" : 0.114269 }
, { "name" : "/model.28/Neg_output_0 copy", "timeMs" : 0.004096, "averageMs" : 0.004096, "percentage" : 0.114269 }
, { "name" : "/model.28/Slice_3_output_0 copy", "timeMs" : 0.004096, "averageMs" : 0.004096, "percentage" : 0.114269 }
, { "name" : "PWN(/model.28/Sigmoid)", "timeMs" : 0.004096, "averageMs" : 0.004096, "percentage" : 0.114269 }
, { "name" : "/model.28/Transpose_2", "timeMs" : 0.005152, "averageMs" : 0.005152, "percentage" : 0.143729 }
, { "name" : "/model.28/Constant_34_output_0 + (Unnamed Layer* 681) [Shuffle]", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/Constant_29_output_0 + (Unnamed Layer* 667) [Shuffle]", "timeMs" : 0, "averageMs" : 0, "percentage" : 0 }
, { "name" : "/model.28/Expand", "timeMs" : 0.019456, "averageMs" : 0.019456, "percentage" : 0.54278 }
, { "name" : "/model.28/Tile", "timeMs" : 0.021504, "averageMs" : 0.021504, "percentage" : 0.599914 }
, { "name" : "PWN(/model.28/Add, /model.28/Mul)", "timeMs" : 0.013312, "averageMs" : 0.013312, "percentage" : 0.371376 }
, { "name" : "/model.28/Transpose_1", "timeMs" : 0.006144, "averageMs" : 0.006144, "percentage" : 0.171404 }
, { "name" : "/model.28/EfficientNMS_TRT", "timeMs" : 0.149504, "averageMs" : 0.149504, "percentage" : 4.17083 }
, { "name" : "Reformatting CopyNode for Output Tensor 2 to /model.28/EfficientNMS_TRT", "timeMs" : 0.003072, "averageMs" : 0.003072, "percentage" : 0.085702 }
, { "name" : "Reformatting CopyNode for Output Tensor 1 to /model.28/EfficientNMS_TRT", "timeMs" : 0.004096, "averageMs" : 0.004096, "percentage" : 0.114269 }
]

# Sort the list by percentage in descending order
sorted_data = sorted(data, key=lambda x: x['percentage'], reverse=True)

# Print sorted list
for item in sorted_data:
    print(item)