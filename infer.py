import numpy as np
import onnxruntime as rt


sess = rt.InferenceSession("translate.onnx")
encoder_input_ids = sess.get_inputs()[0].name
encoder_mask = sess.get_inputs()[1].name
decoder_input_ids = sess.get_inputs()[2].name
decoder_mask = sess.get_inputs()[3].name

pred_onx = sess.run(None, 
    {
        encoder_input_ids: np.zeros((1,512), dtype=np.int64),
        encoder_mask: np.zeros((1,512), dtype=np.int64),
        decoder_input_ids: np.zeros((1,512), dtype=np.int64),
        decoder_mask: np.zeros((1,512), dtype=np.int64),
    }
    )[0]
print(pred_onx)