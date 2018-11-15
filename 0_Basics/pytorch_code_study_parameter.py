import torch


class Parameter(torch.Tensor):
    '''所有参数都会被定义为Parameter类的对象
    
    __new__()方法：在实例化一个类时，先调用__new__()再调用__init__()
                参考：http://www.cnblogs.com/ifantastic/p/3175735.html
                __new__()决定用什么类来实例化一个对象
                __new__(cls,)
    __repr__()方法：在命令行下输出对象名时自动调用，内部调用父类的__repr__()，即tensor的方法
    __reduce_ex__()方法：
    '''
    
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
    """
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __repr__(self):
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()

    def __reduce_ex__(self, proto):
        return Parameter, (super(Parameter, self), self.requires_grad)
