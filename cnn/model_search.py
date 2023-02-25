import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module): # 将节点之间的算子加权平均起来,对应softmax公式

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:  # PRIMITIVES是个list定义了搜索空间的算子类型的名称
      op = OPS[primitive](C, stride, False) # OPS是个字典，key是PRIMITIVES list中的算子名，调用value对应的算子类来实例化一个算子对象
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False)) # OPS中对应的pool类是nn.module的类，不带BN
      self._ops.append(op)
      # 注意以上 affine 都是 False，BN都固定可学习参数
  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops)) # 将所有算子的结构乘上权重求和


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    # preprocess0处理k-2层的输出
    if reduction_prev:  
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False) # 若k-1层是reduction层，要调用FactorizedReduce对k-2层的H×W各降一半
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False) # k-1层不是reduction层，直接用普通的ReLUConvBN融合层保证空间解析度不变
    
    # preprocess1处理k-1层的输出
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps # 一个cell中有4个nodes
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps): # (0-3)共四个中间节点
      for j in range(2+i):  # 每个中间节点与前面所有中间节点和输入节点建立连接，e.g.第0节点与s1和s2输入节点有两天连接，第1节点与s1，s2和0节点有三条连接
        stride = 2 if reduction and j < 2 else 1 # 原文指ajacent to the input nodes为stride2，也就是0号中间节点
        op = MixedOp(C, stride) # 加权求和两个节点之间所有的算子输出
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0) # k-2层
    s1 = self.preprocess1(s1) # k-1层 input

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states)) # self._ops是个list，存放着0-3共4个中间节点的所有算子[0节点与k-1的加权算子，0与k-2的加权算子，1vsk-1，1vsk-2，1vs0，...]
      offset += len(states) # 0节点的offet为0，处理完k-1，k-2后，1节点的offset变为2
      states.append(s)
    
    return torch.cat(states[-self._multiplier:], dim=1) # states是一个存储了前面6个节点（2input node和4中间节点）输出的list，拼接后4（-self._multiplier = -4）个特征向量


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C # 通道缩放系数
    self.stem = nn.Sequential(      # 网络的最初输入: 3通道如，缩放过的通道数出，3×3卷积核带一个BN
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers): # 创建layers个数目的cells构成network
      if i in [layers//3, 2*layers//3]: #网络的1/3和2/3处是reduction层
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]    # 可迭代的对象用 +=等价于加到list中，[cell]去掉[]表示不可迭代的对象，只能用append => 将
      C_prev_prev, C_prev = C_prev, multiplier*C_curr # 更改将k-2层的指针前移一层
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data) # 保留一个模型的结构参数，创建一个新的模型
    return model_new

  def forward(self, input):
    inter_feature = []
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1) # 结构参数是在__init__中调用私有方法_initialize_alphas初始化的，在这里付给weights后传给cells中进行加权
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      if cell.reduction:
        inter_feature.append(s1.clone().detach().cpu())
      s0, s1 = s1, cell(s0, s1, weights)  # s0，s1指针挪一位
    inter_feature.append(s1.clone().detach().cpu())
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, inter_feature

  def _loss(self, input, target):
    logits, _ = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i)) # i节点有i+2个与i前全部节点的连接，k为cell中全部节点的连接数 k =14
    num_ops = len(PRIMITIVES)   # num_ops = 8

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) # 注意，所有的normal cells共享一组参数
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) # 注意，所有的reduction cells共享一组参数
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights): # 输入的是reduction或normal的结构参数
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()   # weights.shape = [14,8]，4个节点与前两个输入节点的连接共14个，8个cells
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2] # 一个节点保留最大的前两个连接
        # k是一个节点一个连接中算子的index, max是将一个连接中算子的结构权重找最大的，除去zero操作，乘上负数，按照负数的升序排列，将最大的系数放在列表的前面，保留前两个
        # 思路是 ：k节点有多个与k之前节点的连接，以每一个连接中全部算子的结构权重里最大的一个排序，保留最大的前两条连接作为edges进入下一个for循环
        
        # 这一个循环找到保留的边权重最大的算子
        for j in edges: # 遍历上一步筛选出来的连接边
          k_best = None
          for k in range(len(W[j])): # 遍历每条边中的算子的权重
            if k != PRIMITIVES.index('none'): # 不考虑zero
              if k_best is None or W[j][k] > W[j][k_best]: # 找算子对应权重最大的算子赋值为k_best
                k_best = k
          gene.append((PRIMITIVES[k_best], j)) #gene存储的是保留下来的算子的名称，以及对应的边的序号
        
        start = end # 通过start和end flag截取list每个节点的
        n += 1

      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()) # 将结构参数a softmax化后分割
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
    
    concat = range(2+self._steps-self._multiplier, self._steps+2) # 指定一个cell中所有节点输出feature map的拼接范围（一般是拼接后4个节点的，也就是从2+self._steps-self._multiplier开始拼接）
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

