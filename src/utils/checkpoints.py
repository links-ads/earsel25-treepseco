from pytorch_lightning.callbacks import ModelCheckpoint
from typing import List

class PartialModelCheckpoint(ModelCheckpoint):
    """This class save only the model weights that contains the save_key_prefix
    in the key. It is possible to decide to include the root of the key or not,
    
    Example:
        save_key_prefix = 'main_model.cls_head'
        ___
        rmv_root = False
        key in the saved model -> main_model.cls_head.mlp0.weight 
        
        rmv_root = True
        key in the saved model -> mlp0.weight
    
    Parameters
    ----------
    ModelCheckpoint : _type_
        _description_
    """
    def __init__(self, *args, key_to_save_prefix: str, key_to_exclude_prefixes: List[str]=None, rmv_root: bool=False, **kwargs):
        """
        Parameters
        ----------
        key_to_save_prefix : str
            Save everything that comes under this prefix
        key_to_exclude_prefixes : List[str]
            Do not save anything that comes under these prefixes
        rmv_root : str
            Remove the root of the key or not (start from the save_key_prefix (included) or from root)
        """
        super().__init__(*args, **kwargs)
        
        self.key_to_save_prefix = key_to_save_prefix if key_to_save_prefix[-1] != '.' else key_to_save_prefix[:-1]
        self.rmv_root = rmv_root
        self.key_to_exclude_prefixes = key_to_exclude_prefixes if key_to_exclude_prefixes is not None else []
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # prefix = 'sam_point_decoder.'
        filtered_state_dict = {}
        
        for key in pl_module.state_dict().keys():
            
            #check if current key has to be saved
            if self.key_to_save_prefix in key:
                # check if key has to be excluded by key_to_exclude_prefixes 
                skip_key = False
                for key_to_exclude in self.key_to_exclude_prefixes:
                    if key_to_exclude in key:
                        skip_key = True
                        break
                if skip_key:
                    continue
                
                # rename key if rmv_root is True
                if self.rmv_root:
                    new_root = self.key_to_save_prefix.split('.')[-1] # never . at the end
                    key_parts = key.split('.')
                    new_key = '.'.join(key_parts[key_parts.index(new_root) + 1:]) # in case of multiple equal sub key take the largest
                else:
                    new_key = key
                filtered_state_dict[new_key] = pl_module.state_dict()[key]
        
        checkpoint['state_dict'] = filtered_state_dict