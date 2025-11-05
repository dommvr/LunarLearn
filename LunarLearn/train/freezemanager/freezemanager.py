class FreezeManager:
    """
    Dynamically freeze/unfreeze layers or parameters during training.

    Example:
        fm = FreezePolicyManager(model)
        fm.add("encoder", freeze_until=5)         # freeze 'encoder' until epoch 5
        fm.add("head", unfreeze_at=10)            # unfreeze 'head' at epoch 10
        fm.add("layer3", freeze_from=20)          # re-freeze 'layer3' at epoch 20

    Then call in Trainer:
        fm.step(epoch=current_epoch)
    """

    def __init__(self, model):
        self.model = model
        self.policies = []

    def add(self, target_name, freeze_until=None, unfreeze_at=None, freeze_from=None):
        """
        Register a freeze/unfreeze policy.

        Args:
            target_name (str): Name of the layer/module to target (must exist in model).
            freeze_until (int, optional): Keep frozen until this epoch.
            unfreeze_at (int, optional): Unfreeze starting from this epoch.
            freeze_from (int, optional): Re-freeze starting from this epoch.
        """
        self.policies.append({
            "name": target_name,
            "freeze_until": freeze_until,
            "unfreeze_at": unfreeze_at,
            "freeze_from": freeze_from,
            "active_state": None,  # track last state to avoid redundant prints
        })

    def _get_target(self, name):
        """Find layer/module by name inside model hierarchy."""
        parts = name.split(".")
        obj = self.model
        for p in parts:
            if not hasattr(obj, p):
                raise AttributeError(f"Layer '{name}' not found in model.")
            obj = getattr(obj, p)
        return obj

    def _apply_freeze(self, target, freeze: bool):
        """Recursively freeze/unfreeze parameters in the target."""
        for p in target.parameters():
            p.frozen = freeze
            if freeze:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def step(self, epoch=None, batch=None):
        """Evaluate and apply freeze/unfreeze policies."""
        for pol in self.policies:
            target = self._get_target(pol["name"])
            changed = False

            # 1. Freeze until
            if pol["freeze_until"] is not None and epoch < pol["freeze_until"]:
                if pol["active_state"] != "frozen":
                    self._apply_freeze(target, True)
                    pol["active_state"] = "frozen"
                    changed = True

            # 2. Unfreeze at
            elif pol["unfreeze_at"] is not None and epoch >= pol["unfreeze_at"]:
                if pol["active_state"] != "unfrozen":
                    self._apply_freeze(target, False)
                    pol["active_state"] = "unfrozen"
                    changed = True

            # 3. Freeze from
            elif pol["freeze_from"] is not None and epoch >= pol["freeze_from"]:
                if pol["active_state"] != "frozen":
                    self._apply_freeze(target, True)
                    pol["active_state"] = "frozen"
                    changed = True

            if changed:
                print(f"[FreezePolicy] '{pol['name']}' set to {pol['active_state']} at epoch {epoch}.")
