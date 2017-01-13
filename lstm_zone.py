from collections import namedtuple
import mxnet as mx


LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias", "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(prefix, num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name=prefix + "t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name=prefix + "t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name=prefix + "t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)

 def lstm_zoneout(prefix, num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0., c_zoneout=0., h_zoneout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name=prefix + "t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name=prefix + "t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name=prefix + "t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")

    
    next_c = mx.sym.Dropout(data=next_c-prev_state.c, p=1-c_zoneout)
    next_c = (1-c_zoneout)*next_c+prev_state.c
    next_h = mx.sym.Dropout(data=next_h-prev_state.h, p=1-h_zoneout)
    next_h = (1-h_zoneout)*next_h+prev_state.h
    return LSTMState(c=next_c, h=next_h)        


def lstm_unroll(
    prefix,
    data,
    num_rnn_layer,
    seq_len,
    num_hidden,
    dropout=0.,
    c_zoneout,
    h_zoneout
):

    prefix += 'rnn_'

    param_cells = []
    last_states = []
    for i in range(num_rnn_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable(prefix + "l%d_i2h_weight" % i),
                                      i2h_bias=mx.sym.Variable(prefix + "l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable(prefix + "l%d_h2h_weight" % i),
                                      h2h_bias=mx.sym.Variable(prefix + "l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable(prefix + "l%d_init_c" % i, attr={'lr_mult': '0'}),
                          h=mx.sym.Variable(prefix + "l%d_init_h" % i, attr={'lr_mult': '0'}))
        last_states.append(state)
    assert(len(last_states) == num_rnn_layer)

    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]

        # stack LSTM
        for i in range(num_rnn_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm_zoneout(prefix, num_hidden, indata=hidden,
                                   prev_state=last_states[i],
                                   param=param_cells[i],
                                   seqidx=seqidx, layeridx=i, dropout=dp_ratio, c_zoneout=c_zoneout, h_zoneout=h_zoneout)
            hidden = next_state.h
            last_states[i] = next_state

        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    net = hidden_all
    return net
