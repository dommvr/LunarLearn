class DeepSupervision:
    def __init__(self, global_weight=1.0, global_scheduler=None):
        self.classifiers = []
        self.global_weight = global_weight
        self.global_scheduler = global_scheduler

    def __call__(self, activations, y, return_each_loss=True):
        return self.forward(activations, y, return_each_loss)
    
    def add(self, aux_classifier):
        if isinstance(aux_classifier, list):
            self.classifiers.extend(aux_classifier)
        else:
            self.classifiers.append(aux_classifier)

    def forward(self, activations, y, return_each_loss=True):
        losses = {}
        total_loss = 0
        for clas in self.classifiers:
            loss = clas(activations, y)
            total_loss += loss

            if return_each_loss:
                losses[clas.level] = loss

        return (total_loss, losses) if return_each_loss else total_loss