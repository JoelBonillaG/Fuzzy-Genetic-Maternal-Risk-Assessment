type TrapezoidPoints = [number, number, number, number];

type MembershipPointSource =
  | TrapezoidPoints
  | { puntos_base?: number[]; puntos_optimizados?: number[]; puntos?: number[] };

export function resolveMembershipPoints(points: MembershipPointSource): TrapezoidPoints {
  if (Array.isArray(points)) {
    return points as TrapezoidPoints;
  }

  const candidate = points.puntos_optimizados ?? points.puntos_base ?? points.puntos;
  if (Array.isArray(candidate) && candidate.length >= 4) {
    return candidate.slice(0, 4) as TrapezoidPoints;
  }

  throw new TypeError("points is not iterable");
}

export function trapezoidMembership(
  x: number,
  points: MembershipPointSource,
) {
  const [a, b, c, d] = resolveMembershipPoints(points);

  if (x < a || x > d) {
    return 0;
  }

  if (a === b && x >= a && x <= c) {
    return 1;
  }

  if (c === d && x >= b && x <= d) {
    return 1;
  }

  if (x >= b && x <= c) {
    return 1;
  }

  if (x > a && x < b) {
    return (x - a) / (b - a);
  }

  if (x > c && x < d) {
    return (d - x) / (d - c);
  }

  return 0;
}

export function generateMembershipSeries(
  domain: [number, number],
  points: MembershipPointSource,
  steps = 140,
) {
  const [min, max] = domain;
  const step = (max - min) / (steps - 1);

  return Array.from({ length: steps }, (_, index) => {
    const x = Number((min + step * index).toFixed(2));
    return {
      x,
      membership: Number(trapezoidMembership(x, points).toFixed(4)),
    };
  });
}
