from .Naive import load_naive_model

from .Hierarchial import load_hierarchial_model

from .Graph import load_graph_model


model_loader_dict = {"naive":load_naive_model,"hierarchial":load_hierarchial_model,"graph":load_graph_model}
# __all__ = [model_loader_dict]