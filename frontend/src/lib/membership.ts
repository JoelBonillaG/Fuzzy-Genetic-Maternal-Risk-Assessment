export function trapezoidMembership(
  x: number,
  points: [number, number, number, number],
) {
  const [a, b, c, d] = points;

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
  points: [number, number, number, number],
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
