import tensorflow as tf
import resnet

def get_model(model_name, num_classes=10, weight_decay=0.0):
    if model_name.lower()=='resnet18':
        return resnet.ResNet18(num_classes=num_classes, weight_decay=weight_decay)
    elif model_name.lower()=='resnet34':
        return resnet.ResNet34(num_classes=num_classes, weight_decay=weight_decay)
    elif model_name.lower()=='resnet50':
        return resnet.ResNet50(num_classes=num_classes, weight_decay=weight_decay)
    elif model_name.lower()=='resnet101':
        return resnet.ResNet18(num_classes=num_classes, weight_decay=weight_decay)
    elif model_name.lower()=='resnet152':
        return resnet.ResNet152(num_classes=num_classes, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown model name {}".format(model_name)) 

    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
