from .davidon_fletcher_powell import davidon_fletcher_powell
from .fletcher_reeves import fletcher_reeves
from .gradient_projection import gradient_projection
from .zoutendijk import zoutendijk

def minimize(
    fun,
    x0,
    jac,
    method,
    callback,
    options,
):
    if method == "davidon_fletcher_powell":
        return davidon_fletcher_powell(
            fun,
            x0,
            jac,
            callback,
            options,
        )
    elif method == "fletcher_reeves":
        return fletcher_reeves(
            fun,
            x0,
            jac,
            callback,
            options,
        )
    elif method == "gradient_projection":
        return gradient_projection(
            fun,
            x0,
            jac,
            callback,
            options,
        )
    elif method == "zoutendijk":
        return zoutendijk(
            fun,
            x0,
            jac,
            callback,
            options,
        )