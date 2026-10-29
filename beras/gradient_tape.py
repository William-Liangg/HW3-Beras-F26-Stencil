from collections import defaultdict

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs and return the gradients of the sources

        q = [target]                    ## Live q; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)  ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful

        # bfs traversal, iterative with q data struct
        while q:
            current_tensor = q.pop(0)

            layer = self.previous_layers[id(current_tensor)]
            if layer is None:
                continue

            current_grad = grads[id(current_tensor)]

            # only compute weight gradients if the layer actually has weights
            if layer.trainable_variables:
                grad_lay_weights = layer.compose_weight_gradients(current_grad)
                for weight, grad in zip(layer.trainable_variables, grad_lay_weights):
                    grads[id(weight)] = [grad]

            # layer inputs
            grad_lay_input = layer.compose_input_gradients(current_grad)
            for input_tensor, grad in zip(layer.inputs, grad_lay_input):
                grads[id(input_tensor)] = [grad]
                q.append(input_tensor)


        res = []
        for s in sources:
            source_grads = grads[id(s)]
            if source_grads is None:
                res.append(None)
            else:
                res.append(source_grads[0])
        return res
            