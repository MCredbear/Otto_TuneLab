from typing import List

def gender_from_pitch_derivative(pitch_points: List[float], scale: float, clamp: float = 1.0) -> List[float]:
    """
    pitch_points: [t0,p0,t1,p1,...]  (t是tick, p是midi)
    Gender values: 同样 [t0,g0,t1,g1,...]
    g = scale * dp/dt  (dt用tick)
    """
    if len(pitch_points) < 4:
        return []
    out = []
    prev_t = float(pitch_points[0])
    prev_p = float(pitch_points[1])
    out.extend([prev_t, 0.0])
    for k in range(2, len(pitch_points), 2):
        t = float(pitch_points[k])
        p = float(pitch_points[k+1])
        dt = t - prev_t
        g = scale * ((p - prev_p) / dt)
        if clamp is not None:
            g = max(-clamp, min(clamp, g))
        out.extend([t, abs(float(g))])
        prev_t, prev_p = t, p
    return out

def breathiness_from_pitch_derivative(pitch_points: List[float], scale: float, clamp: float = 1.0) -> List[float]:
    """
    pitch_points: [t0,p0,t1,p1,...]  (t是tick, p是midi)
    Breathiness values: 同样 [t0,g0,t1,g1,...]
    g = scale * dp/dt  (dt用tick)
    """
    if len(pitch_points) < 4:
        return []
    out = []
    prev_t = float(pitch_points[0])
    prev_p = float(pitch_points[1])
    out.extend([prev_t, 0.0])
    for k in range(2, len(pitch_points), 2):
        t = float(pitch_points[k])
        p = float(pitch_points[k+1])
        dt = t - prev_t
        g = abs(scale * ((p - prev_p) / dt))
        if clamp is not None:
            g = min(clamp, max(0, g))
        out.extend([t, float(g)])
        prev_t, prev_p = t, p
    return out