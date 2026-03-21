import { describe, it, expect } from 'vitest';
import {
  partitionBatnaWatna,
  computeZopa,
  computeNegotiationDynamics,
  colorByNegotiationRole,
  computeReservationSurface,
} from './voidmap-negotiation.js';
import { createVoidMap, recordFrame } from './voidmap.js';
import type { TopologyBranch } from './index.js';

function makeBranch(id: string, cost: number, quality: number): TopologyBranch {
  return { id, vertices: 100, drawCalls: 1, estimatedCostMs: cost, quality };
}

function makeFrame(branches: TopologyBranch[], ventedIds: string[]) {
  const surviving = branches.filter((b) => !ventedIds.includes(b.id));
  return {
    input: { branches, strategy: 'race' as const, budgetMs: 8 },
    result: {
      collapsedBy: 'race' as const, winnerId: surviving[0]?.id ?? null,
      survivingBranches: surviving, ventedBranchIds: ventedIds,
      totalVertices: 0, totalDrawCalls: 0, totalCostMs: 0, budgetMs: 8, overBudget: false,
    },
  };
}

function buildAsymmetricMap() {
  // A rarely fails (BATNA), B sometimes fails, C always fails (WATNA)
  const branches = [makeBranch('A', 3, 0.9), makeBranch('B', 6, 0.5), makeBranch('C', 10, 0.2)];
  let map = createVoidMap();
  for (let i = 0; i < 100; i++) {
    let ventedId: string;
    if (i % 10 < 1) ventedId = 'A';      // 10% failure
    else if (i % 10 < 4) ventedId = 'B'; // 30% failure
    else ventedId = 'C';                  // 60% failure
    const f = makeFrame(branches, [ventedId]);
    map = recordFrame(map, f.input, f.result);
  }
  return map;
}

describe('partitionBatnaWatna', () => {
  it('classifies branches into BATNA and WATNA', () => {
    const map = buildAsymmetricMap();
    const partition = partitionBatnaWatna(map);

    // A has fewest rejections → BATNA
    // C has most rejections → WATNA
    const batnaIds = partition.batna.map((a) => a.branchId);
    const watnaIds = partition.watna.map((a) => a.branchId);
    expect(batnaIds).toContain('A');
    expect(watnaIds).toContain('C');
  });

  it('computes BATNA strength and WATNA severity', () => {
    const map = buildAsymmetricMap();
    const partition = partitionBatnaWatna(map);
    expect(partition.batnaStrength).toBeGreaterThan(0);
    expect(partition.watnaSeverity).toBeGreaterThan(0);
  });

  it('computes Lorentz scalar', () => {
    const map = buildAsymmetricMap();
    const partition = partitionBatnaWatna(map);
    expect(typeof partition.lorentzScalar).toBe('number');
    expect(['space-like', 'light-like', 'time-like']).toContain(partition.viability);
  });

  it('returns empty for empty map', () => {
    const partition = partitionBatnaWatna(createVoidMap());
    expect(partition.batna).toHaveLength(0);
    expect(partition.watna).toHaveLength(0);
    expect(partition.viability).toBe('time-like');
  });
});

describe('computeZopa', () => {
  it('finds overlap between two parties', () => {
    // Party A mostly rejects C, party B mostly rejects A
    const branches = [makeBranch('A', 3, 0.9), makeBranch('B', 6, 0.5), makeBranch('C', 10, 0.2)];
    let mapA = createVoidMap();
    let mapB = createVoidMap();

    for (let i = 0; i < 50; i++) {
      const ventedA = i % 5 < 4 ? 'C' : 'A';
      mapA = recordFrame(mapA, makeFrame(branches, [ventedA]).input, makeFrame(branches, [ventedA]).result);

      const ventedB = i % 5 < 4 ? 'A' : 'C';
      mapB = recordFrame(mapB, makeFrame(branches, [ventedB]).input, makeFrame(branches, [ventedB]).result);
    }

    // Use low eta + threshold since high eta collapses complement weights to near-zero
    const zopa = computeZopa(mapA, mapB, { eta: 0.05, zopaThreshold: 0.01 });
    // B should be in ZOPA (neither party strongly rejects it)
    expect(zopa.zopaRegion.length).toBeGreaterThan(0);
    expect(zopa.zopaExists).toBe(true);
    expect(zopa.nadirBranch).toBeTruthy();
  });

  it('computes mutual information', () => {
    const map = buildAsymmetricMap();
    const zopa = computeZopa(map, map);
    expect(zopa.mutualInformation).toBeGreaterThanOrEqual(0);
  });

  it('returns empty ZOPA for empty maps', () => {
    const zopa = computeZopa(createVoidMap(), createVoidMap());
    expect(zopa.zopaExists).toBe(false);
  });
});

describe('computeNegotiationDynamics', () => {
  it('tracks Lorentz scalar over time', () => {
    const map = buildAsymmetricMap();
    const dynamics = computeNegotiationDynamics(map);
    expect(dynamics.lorentzHistory.length).toBeGreaterThan(0);
    expect(['exploring', 'narrowing', 'converging', 'deadlocked', 'walk-away']).toContain(dynamics.phase);
  });

  it('detects momentum', () => {
    const map = buildAsymmetricMap();
    const dynamics = computeNegotiationDynamics(map);
    expect(typeof dynamics.momentum).toBe('number');
  });
});

describe('colorByNegotiationRole', () => {
  it('assigns colors to all tombstones', () => {
    const map = buildAsymmetricMap();
    const colored = colorByNegotiationRole(map);
    expect(colored.length).toBe(map.tombstones.length);

    for (const ct of colored) {
      expect(['batna', 'watna', 'marginal']).toContain(ct.classification);
      expect(ct.color.r).toBeGreaterThanOrEqual(0);
      expect(ct.color.r).toBeLessThanOrEqual(1);
    }
  });

  it('WATNA tombstones are red-ish, BATNA are blue-ish', () => {
    const map = buildAsymmetricMap();
    const colored = colorByNegotiationRole(map);

    const watnaTombstones = colored.filter((c) => c.classification === 'watna');
    const batnaTombstones = colored.filter((c) => c.classification === 'batna');

    if (watnaTombstones.length > 0) {
      const avgR = watnaTombstones.reduce((s, c) => s + c.color.r, 0) / watnaTombstones.length;
      expect(avgR).toBeGreaterThan(0.5); // Red-ish
    }

    if (batnaTombstones.length > 0) {
      const avgB = batnaTombstones.reduce((s, c) => s + c.color.b, 0) / batnaTombstones.length;
      expect(avgB).toBeGreaterThan(0.3); // Blue-ish
    }
  });
});

describe('computeReservationSurface', () => {
  it('produces boundary points', () => {
    const map = buildAsymmetricMap();
    // Use low eta so complement weights don't collapse to zero
    const surface = computeReservationSurface(map, { eta: 0.01 });
    expect(surface.boundaryPoints.length).toBeGreaterThan(0);
    expect(surface.reservationWeight).toBeGreaterThanOrEqual(0);
  });

  it('returns empty for empty map', () => {
    const surface = computeReservationSurface(createVoidMap());
    expect(surface.boundaryPoints).toHaveLength(0);
  });
});
