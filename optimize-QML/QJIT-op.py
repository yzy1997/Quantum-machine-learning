import pennylane as qml
from jax import numpy as jnp
# import numpy as np
import optax
import catalyst
import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


n_wires = 5
# data = np.sin(np.mgrid[-2:2:0.2].reshape(n_wires, -1)) ** 3
data = jnp.sin(jnp.mgrid[-2:2:0.2].reshape(n_wires, -1)) ** 3
targets = jnp.array([-0.2, 0.4, 0.35, 0.2])

dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev)
def circuit(data, weights):
    """Quantum circuit ansatz"""

    @qml.for_loop(0, n_wires, 1)
    def data_embedding(i):
        qml.RY(data[i], wires=i)

    data_embedding()

    @qml.for_loop(0, n_wires, 1)
    def ansatz(i):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RX(weights[i, 2], wires=i)
        qml.CNOT(wires=[i, (i + 1) % n_wires])

    ansatz()

    # we use a sum of local Z's as an observable since a
    # local Z would only be affected by params on that qubit.
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

circuit = qml.qjit(catalyst.vmap(circuit, in_axes=(1, None)))

def my_model(data, weights, bias):
    return circuit(data, weights) + bias

@qml.qjit
def loss_fn(params, data, targets):
    predictions = my_model(data, params["weights"], params["bias"])
    loss = jnp.sum((targets - predictions) ** 2 / len(data))
    return loss

weights = jnp.ones([n_wires, 3])
bias = jnp.array(0.)
params = {"weights": weights, "bias": bias}

loss_fn(params, data, targets)

print(qml.qjit(catalyst.grad(loss_fn, method="fd"))(params, data, targets))