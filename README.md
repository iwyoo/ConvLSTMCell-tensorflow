# ConvLSTMCell-tensorflow
Convolutional LSTM network cell (ConvLSTM).
The implementation is based on (http://arxiv.org/abs/1506.04214) and BasicLSTMCell in TensorFlow.
 
## Example
```python
p_input = tf.placeholder(tf.float32, [None, height, width, step_size, channel])
p_label = tf.placeholder(tf.float32, [None, height, width, 3])

p_input_list = tf.split(3, step_size, p_input)
p_input_list = [tf.squeeze(p_input_, [3]) for p_input_ in p_input_list]

cell = ConvLSTMCell(hidden_num)
state = cell.zero_state(batch_size, height, width)

with tf.variable_scope("ConvLSTM") as scope: # as BasicLSTMCell
  for i, p_input_ in enumerate(p_input_list):
    if i > 0: 
      scope.reuse_variables()
    # ConvCell takes Tensor with size [batch_size, height, width, channel].
    t_output, state = cell(p_input_, state)
```
