import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(), ##### 指定优化结构参数
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  
  # target：以一阶还是二阶估计计算结构参数在验证集上的梯度
  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid) 
    
    self.optimizer.step() # 上述反传后，更新结构参数的梯度

  # target：一阶估计 → 当sigma=0时，为一阶估计梯度，即用w估计w*的梯度，计算Lval(w,a)对a的梯度
  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid) # model._loss在model_search.py的Network类中定义，用于计算网络的loss，_loss函数包含 1)logits = model(inputs)；2）loss = criterion(logits,targets)
    loss.backward()

  # target：二阶梯度估计 → 更新各个参数的梯度
  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer) # 在训练集计算w` = w - xigma*Ltrain(w,a)，以w`创建新模型
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward() # 以w`创建的模型计算一次前向
    dalpha = [v.grad for v in unrolled_model.arch_parameters()] # 提取结构参数的梯度
    vector = [v.grad.data for v in unrolled_model.parameters()] # 提取权重在val上的梯度 ##### 注意这里unrolled模型的使命到此为止，其模型仅仅是为了计算w`，提取出参数到vector和dalpha后放弃
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads): # w* 的梯度减去二阶梯度估计后，得到新的梯度
      g.data.sub_(other = ig.data, alpha = eta)

    for v, g in zip(self.model.arch_parameters(), dalpha): # 将上一个循环得到新的梯度赋值给现在的梯度，类似于backward()
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
  
  # target：手动计算一次w在train上的梯度，计算w` = w - xigma*Ltrain(w,a)
  # return：一个以w`创建的新模型，a不变
  def _compute_unrolled_model(self, input, target, eta, network_optimizer): 
    loss = self.model._loss(input, target)  # 计算train上的loss
    theta = _concat(self.model.parameters()).data # 将模型的权重拼接为1行后，用.data取出tensor本体数据，舍弃grad, grad_fn等额外的反向图计算过程需要的信息
    try:
      # optimizer.state 是一个字典，key是网络的权重，value是上一步的动量，本行是更新动量
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum) # 从optimizer的缓存中提取上一步的动量，乘上动量系数构成这一步的动量
    except:
      moment = torch.zeros_like(theta)

    # torch.autograd.grad是计算形参第一个变量对第二个变量的梯度和
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta # 总梯度：weight_decay×权重 + loss对w的梯度 
    # 手动更新好算子的权重参数后，创建一个保留结构参数的新的模型
    unrolled_model = self._construct_model_from_theta(theta.sub(other=moment+dtheta, alpha = eta)) # sub(eta, moment+dtheta)更改为sub(other,*,alpha) → w` = w -signma(moment+J_train(loss对w的梯度) + weight_decay*w)
    return unrolled_model

  # target: 创建一个结构不变的模型，其权重是w` = w-sita*L_train(w,a),用这个模型前向一次得到梯度后不在使用
  def _construct_model_from_theta(self, theta):
    model_new = self.model.new() # 创建一个保留结构参数的新模型
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size()) # 计算一层参数的长度
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params) # 将截取的参数放入字典中
    model_new.load_state_dict(model_dict) # 将前向一次的参数赋给新的模型
    return model_new.cuda()

  # target: 计算前向和后向w±模型对结构参数的梯度
  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm() # vector是权重参数的梯度的2范数
    
    # 计算train 上 w+对a的梯度
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(other = v, alpha = R) # p.data in_place加更新模型的参数，即w+ = w+R*v
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters()) # p = positive前向一步

    
    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(other = v, alpha =  2*R) # w- = w-2R*v /v 是w`对a的梯度
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters()) # n = negative，后向一步

    for p, v in zip(self.model.parameters(), vector): # 计算完后恢复梯度（w- + v*R  →  w）
      p.data.add_(other = v, alpha = R)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

