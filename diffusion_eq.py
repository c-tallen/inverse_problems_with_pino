from physicsnemo.sym.eq.pde import PDE
from sympy import Function, Number, Symbol

class Diffusion(PDE):
    """Diffusion equation: ``dT/dt - div(D * grad(T)) = Q``.

    Equivalent to ``physicsnemo-sym``'s ``Diffusion`` class for the 2-D,
    steady-state case with variable diffusivity ``D`` as a SymPy Function.

    Reference: https://en.wikipedia.org/wiki/Diffusion_equation
    """

    def __init__(self, T="T", D="D", Q=0, dim=2, time=False):
        """Initialize with variable name *T*, diffusivity *D*, and source *Q*."""
        self.dim = dim
        x, y = Symbol("x"), Symbol("y")
        iv = {"x": x, "y": y}
        T_var = Function(T)(*iv.values())
        D_var = Function(D)(*iv.values()) if isinstance(D, str) else Number(D)
        Q_var = Number(Q) if isinstance(Q, (int, float)) else Q
        self.equations = {
            f"diffusion_{T}": (
                (T_var.diff(Symbol("t")) if time else 0)
                - (D_var * T_var.diff(x)).diff(x)
                - (D_var * T_var.diff(y)).diff(y)
                - Q_var
            ),
        }


class Burgers1D(PDE):
    """1D viscous Burgers' equation: u_t + u * u_x - nu * u_xx = 0.

    In symbolic form:

        ∂u/∂t + u ∂u/∂x - ν ∂²u/∂x² = 0

    where:
        u  = solution field
        nu = viscosity / diffusion coefficient
    """

    def __init__(self, u="u", nu="nu", time=True):
        """Initialize with solution variable *u* and viscosity *nu*."""
        x = Symbol("x")
        t = Symbol("t")

        iv = {"x": x, "t": t} if time else {"x": x}

        u_var = Function(u)(*iv.values())

        if isinstance(nu, str):
            nu_var = Function(nu)(*iv.values())
        elif isinstance(nu, (int, float)):
            nu_var = Number(nu)
        else:
            nu_var = nu

        self.equations = {
            f"burgers_{u}": (
                (u_var.diff(t) if time else 0)
                + u_var * u_var.diff(x)
                - nu_var * u_var.diff(x, 2)
            )
        }