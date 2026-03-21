/**
 * voidmap-negotiation.ts
 *
 * BATNA/WATNA analysis of the void boundary. The void IS the negotiation
 * surface: tombstones with high rejection counts are the WATNA (worst
 * alternatives to negotiated agreement). The complement distribution peak
 * is the BATNA (best alternative). The Lorentz scalar s² = batna² - watna²
 * determines viability:
 *
 *   space-like (s² > 0): BATNA dominates → room to negotiate
 *   light-like (s² ~ 0): balanced → marginal position
 *   time-like  (s² < 0): WATNA dominates → walk away
 *
 * When two parties' void boundaries are compared, the ZOPA (zone of
 * possible agreement) is where their complement distributions overlap.
 */

import type { Vec3 } from './index.js';
import { addVec3, scaleVec3, subtractVec3, lengthVec3 } from './index.js';
import type { VoidMap, Tombstone, VoidBoundaryStats } from './voidmap.js';
import { complementDistribution, voidEntropy, computeVoidBoundary } from './voidmap.js';

// ---------------------------------------------------------------------------
// BATNA/WATNA partition
// ---------------------------------------------------------------------------

export interface NegotiationAlternative {
  branchId: string;
  rejectionCount: number;
  complementWeight: number;
  averageCost: number;
  averageQuality: number;
  /** Position centroid of this branch's tombstones */
  centroid: Vec3;
  /** Classification */
  classification: 'batna' | 'watna' | 'marginal';
}

export interface BatnaWatnaPartition {
  /** Best alternatives (low rejection, high complement weight) */
  batna: readonly NegotiationAlternative[];
  /** Worst alternatives (high rejection, low complement weight) */
  watna: readonly NegotiationAlternative[];
  /** Marginal alternatives (near the threshold) */
  marginal: readonly NegotiationAlternative[];
  /** BATNA strength: weighted average of complement weights for BATNA branches */
  batnaStrength: number;
  /** WATNA severity: weighted average of rejection rates for WATNA branches */
  watnaSeverity: number;
  /** Lorentz settlement score: s² = batna² - watna² */
  lorentzScalar: number;
  /** Viability classification */
  viability: 'space-like' | 'light-like' | 'time-like';
  /** Reservation price: the complement weight at the BATNA/WATNA boundary */
  reservationWeight: number;
  /** Walk-away threshold: rejection count above which alternatives are WATNA */
  walkAwayThreshold: number;
}

/**
 * Partition the void boundary into BATNA and WATNA regions.
 *
 * The threshold is computed as the median rejection count. Branches
 * rejected less than median are BATNA (they survived more often).
 * Branches rejected more than median are WATNA (they failed more often).
 * Branches within 10% of median are marginal.
 */
export function partitionBatnaWatna(
  map: VoidMap,
  options?: { eta?: number; marginPercent?: number }
): BatnaWatnaPartition {
  const eta = options?.eta ?? 3.0;
  const marginPercent = options?.marginPercent ?? 0.1;

  const compDist = complementDistribution(map, eta);
  const branchIds = Array.from(map.rejectionCounts.keys());

  if (branchIds.length === 0) {
    return {
      batna: [], watna: [], marginal: [],
      batnaStrength: 0, watnaSeverity: 0,
      lorentzScalar: 0, viability: 'time-like',
      reservationWeight: 0, walkAwayThreshold: 0,
    };
  }

  // Compute per-branch stats
  const alternatives: NegotiationAlternative[] = [];
  for (let i = 0; i < branchIds.length; i++) {
    const branchId = branchIds[i];
    const rejectionCount = map.rejectionCounts.get(branchId) ?? 0;
    const tombstones = map.tombstones.filter((t) => t.branchId === branchId);

    let centroid: Vec3 = { x: 0, y: 0, z: 0 };
    let totalCost = 0;
    let totalQuality = 0;
    if (tombstones.length > 0) {
      for (const t of tombstones) {
        centroid = addVec3(centroid, t.position);
        totalCost += t.cost;
        totalQuality += t.quality;
      }
      centroid = scaleVec3(centroid, 1 / tombstones.length);
    }

    alternatives.push({
      branchId,
      rejectionCount,
      complementWeight: i < compDist.length ? compDist[i] : 0,
      averageCost: tombstones.length > 0 ? totalCost / tombstones.length : 0,
      averageQuality: tombstones.length > 0 ? totalQuality / tombstones.length : 0,
      centroid,
      classification: 'marginal', // will be set below
    });
  }

  // Compute threshold (median rejection count)
  const counts = alternatives.map((a) => a.rejectionCount).sort((a, b) => a - b);
  const median = counts[Math.floor(counts.length / 2)];
  const marginSize = Math.max(1, Math.ceil(median * marginPercent));
  const walkAwayThreshold = median;

  // Classify
  const batna: NegotiationAlternative[] = [];
  const watna: NegotiationAlternative[] = [];
  const marginal: NegotiationAlternative[] = [];

  for (const alt of alternatives) {
    if (alt.rejectionCount < median - marginSize) {
      alt.classification = 'batna';
      batna.push(alt);
    } else if (alt.rejectionCount > median + marginSize) {
      alt.classification = 'watna';
      watna.push(alt);
    } else {
      alt.classification = 'marginal';
      marginal.push(alt);
    }
  }

  // BATNA strength: weighted average of complement weights
  const batnaStrength = batna.length > 0
    ? batna.reduce((sum, a) => sum + a.complementWeight, 0) / batna.length
    : 0;

  // WATNA severity: weighted average of rejection rates
  const maxRejections = Math.max(...counts, 1);
  const watnaSeverity = watna.length > 0
    ? watna.reduce((sum, a) => sum + a.rejectionCount / maxRejections, 0) / watna.length
    : 0;

  // Lorentz settlement score
  const lorentzScalar = batnaStrength * batnaStrength - watnaSeverity * watnaSeverity;

  // Viability
  let viability: BatnaWatnaPartition['viability'];
  if (lorentzScalar > 0.01) viability = 'space-like';
  else if (lorentzScalar < -0.01) viability = 'time-like';
  else viability = 'light-like';

  // Reservation weight: complement weight at the boundary
  const reservationWeight = marginal.length > 0
    ? marginal.reduce((sum, a) => sum + a.complementWeight, 0) / marginal.length
    : (batnaStrength + (1 - watnaSeverity)) / 2;

  return {
    batna: batna.sort((a, b) => b.complementWeight - a.complementWeight),
    watna: watna.sort((a, b) => a.complementWeight - b.complementWeight),
    marginal,
    batnaStrength: Number(batnaStrength.toFixed(6)),
    watnaSeverity: Number(watnaSeverity.toFixed(6)),
    lorentzScalar: Number(lorentzScalar.toFixed(6)),
    viability,
    reservationWeight: Number(reservationWeight.toFixed(6)),
    walkAwayThreshold,
  };
}

// ---------------------------------------------------------------------------
// ZOPA: zone of possible agreement between two parties
// ---------------------------------------------------------------------------

export interface ZopaAnalysis {
  /** Branches where both parties' complement weights are above threshold */
  zopaRegion: readonly {
    branchId: string;
    weightA: number;
    weightB: number;
    jointWeight: number;
  }[];
  /** Is there a viable ZOPA? */
  zopaExists: boolean;
  /** ZOPA width: fraction of branches in the ZOPA */
  zopaWidth: number;
  /** Nadir point: the branch with highest joint weight (optimal settlement) */
  nadirBranch: string | null;
  nadirJointWeight: number;
  /** Joint entropy of the overlapping distributions */
  jointEntropy: number;
  /** Mutual information between the two distributions */
  mutualInformation: number;
  /** Per-party viability */
  partyAViability: BatnaWatnaPartition['viability'];
  partyBViability: BatnaWatnaPartition['viability'];
}

/**
 * Compute the ZOPA between two parties' void boundaries.
 * The ZOPA is where both parties' complement distributions overlap --
 * alternatives that neither party strongly rejects.
 */
export function computeZopa(
  partyAMap: VoidMap,
  partyBMap: VoidMap,
  options?: { eta?: number; zopaThreshold?: number }
): ZopaAnalysis {
  const eta = options?.eta ?? 3.0;
  const zopaThreshold = options?.zopaThreshold ?? 0.05;

  // Align distributions by branch ID
  const allBranches = Array.from(new Set([
    ...partyAMap.rejectionCounts.keys(),
    ...partyBMap.rejectionCounts.keys(),
  ])).sort();

  if (allBranches.length === 0) {
    return {
      zopaRegion: [], zopaExists: false, zopaWidth: 0,
      nadirBranch: null, nadirJointWeight: 0,
      jointEntropy: 0, mutualInformation: 0,
      partyAViability: 'time-like', partyBViability: 'time-like',
    };
  }

  function computeAligned(map: VoidMap): number[] {
    const counts = allBranches.map((b) => map.rejectionCounts.get(b) ?? 0);
    const logits = counts.map((v) => -eta * v);
    const maxLogit = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }

  const distA = computeAligned(partyAMap);
  const distB = computeAligned(partyBMap);

  // Joint distribution (outer product normalized)
  const joint: number[] = [];
  for (let i = 0; i < allBranches.length; i++) {
    joint.push(distA[i] * distB[i]);
  }
  const jointSum = joint.reduce((a, b) => a + b, 0);
  const normalizedJoint = jointSum > 0
    ? joint.map((j) => j / jointSum)
    : joint;

  // ZOPA region: branches where both weights exceed threshold
  const zopaRegion: ZopaAnalysis['zopaRegion'][number][] = [];
  let nadirBranch: string | null = null;
  let nadirJointWeight = 0;

  for (let i = 0; i < allBranches.length; i++) {
    if (distA[i] >= zopaThreshold && distB[i] >= zopaThreshold) {
      const jointWeight = normalizedJoint[i];
      zopaRegion.push({
        branchId: allBranches[i],
        weightA: Number(distA[i].toFixed(6)),
        weightB: Number(distB[i].toFixed(6)),
        jointWeight: Number(jointWeight.toFixed(6)),
      });
      if (jointWeight > nadirJointWeight) {
        nadirJointWeight = jointWeight;
        nadirBranch = allBranches[i];
      }
    }
  }

  // Joint entropy
  let jointEntropy = 0;
  for (const p of normalizedJoint) {
    if (p > 1e-10) jointEntropy -= p * Math.log2(p);
  }

  // Marginal entropies
  let entropyA = 0;
  let entropyB = 0;
  for (let i = 0; i < allBranches.length; i++) {
    if (distA[i] > 1e-10) entropyA -= distA[i] * Math.log2(distA[i]);
    if (distB[i] > 1e-10) entropyB -= distB[i] * Math.log2(distB[i]);
  }

  // Mutual information: I(A;B) = H(A) + H(B) - H(A,B)
  const mutualInformation = Math.max(0, entropyA + entropyB - jointEntropy);

  // Per-party viability
  const partitionA = partitionBatnaWatna(partyAMap, { eta });
  const partitionB = partitionBatnaWatna(partyBMap, { eta });

  return {
    zopaRegion: zopaRegion.sort((a, b) => b.jointWeight - a.jointWeight),
    zopaExists: zopaRegion.length > 0,
    zopaWidth: allBranches.length > 0
      ? Number((zopaRegion.length / allBranches.length).toFixed(4))
      : 0,
    nadirBranch,
    nadirJointWeight: Number(nadirJointWeight.toFixed(6)),
    jointEntropy: Number(jointEntropy.toFixed(6)),
    mutualInformation: Number(mutualInformation.toFixed(6)),
    partyAViability: partitionA.viability,
    partyBViability: partitionB.viability,
  };
}

// ---------------------------------------------------------------------------
// Negotiation dynamics over time
// ---------------------------------------------------------------------------

export interface NegotiationDynamics {
  /** Lorentz scalar at each round */
  lorentzHistory: readonly { round: number; s2: number; viability: string }[];
  /** Walk-away round: first round where s² goes time-like (if ever) */
  walkAwayRound: number | null;
  /** Convergence round: first round where s² stabilizes space-like */
  convergenceRound: number | null;
  /** Phase: current negotiation phase */
  phase: 'exploring' | 'narrowing' | 'converging' | 'deadlocked' | 'walk-away';
  /** Momentum: consecutive rounds of improving s² */
  momentum: number;
}

/**
 * Compute negotiation dynamics over time by slicing the void at each round
 * and tracking the Lorentz scalar evolution.
 */
export function computeNegotiationDynamics(
  map: VoidMap,
  options?: { eta?: number; sampleInterval?: number }
): NegotiationDynamics {
  const eta = options?.eta ?? 3.0;
  const interval = options?.sampleInterval ?? Math.max(1, Math.floor(map.round / 100));

  const history: { round: number; s2: number; viability: string }[] = [];
  let walkAwayRound: number | null = null;
  let convergenceRound: number | null = null;
  let consecutiveSpaceLike = 0;
  let momentum = 0;
  let prevS2 = 0;

  for (let r = interval; r <= map.round; r += interval) {
    // Build a temporary map sliced to this round
    const filtered = map.tombstones.filter((t) => t.round <= r);
    const counts = new Map<string, number>();
    for (const t of filtered) {
      counts.set(t.branchId, (counts.get(t.branchId) ?? 0) + 1);
    }
    const slicedMap: VoidMap = { tombstones: filtered, round: r, rejectionCounts: counts };

    const partition = partitionBatnaWatna(slicedMap, { eta });
    history.push({
      round: r,
      s2: partition.lorentzScalar,
      viability: partition.viability,
    });

    if (partition.viability === 'time-like' && walkAwayRound === null) {
      walkAwayRound = r;
    }

    if (partition.viability === 'space-like') {
      consecutiveSpaceLike++;
      if (consecutiveSpaceLike >= 5 && convergenceRound === null) {
        convergenceRound = r;
      }
    } else {
      consecutiveSpaceLike = 0;
    }

    if (partition.lorentzScalar > prevS2) {
      momentum++;
    } else {
      momentum = 0;
    }
    prevS2 = partition.lorentzScalar;
  }

  // Determine phase
  let phase: NegotiationDynamics['phase'];
  const latest = history[history.length - 1];
  if (!latest) {
    phase = 'exploring';
  } else if (latest.viability === 'time-like') {
    phase = 'walk-away';
  } else if (convergenceRound !== null && latest.viability === 'space-like') {
    phase = 'converging';
  } else if (momentum === 0 && latest.viability === 'light-like') {
    phase = 'deadlocked';
  } else if (history.length > 5) {
    phase = 'narrowing';
  } else {
    phase = 'exploring';
  }

  return {
    lorentzHistory: history,
    walkAwayRound,
    convergenceRound,
    phase,
    momentum,
  };
}

// ---------------------------------------------------------------------------
// Tombstone coloring by BATNA/WATNA classification
// ---------------------------------------------------------------------------

export interface NegotiationColoredTombstone {
  tombstone: Tombstone;
  classification: 'batna' | 'watna' | 'marginal';
  /** 0-1 intensity: 1 = strong BATNA/WATNA, 0 = marginal */
  intensity: number;
  /** RGB color based on classification */
  color: { r: number; g: number; b: number };
}

/**
 * Color each tombstone by its BATNA/WATNA classification.
 * BATNA = blue-green (survived more), WATNA = red-orange (failed more),
 * Marginal = gray (near the boundary).
 */
export function colorByNegotiationRole(
  map: VoidMap,
  options?: { eta?: number }
): readonly NegotiationColoredTombstone[] {
  const partition = partitionBatnaWatna(map, options);

  // Build lookup
  const classificationMap = new Map<string, { classification: NegotiationAlternative['classification']; complementWeight: number }>();
  for (const alt of [...partition.batna, ...partition.watna, ...partition.marginal]) {
    classificationMap.set(alt.branchId, {
      classification: alt.classification,
      complementWeight: alt.complementWeight,
    });
  }

  const maxWeight = Math.max(
    ...partition.batna.map((a) => a.complementWeight),
    ...partition.watna.map((a) => a.complementWeight),
    0.001
  );

  return map.tombstones.map((t) => {
    const info = classificationMap.get(t.branchId);
    const classification = info?.classification ?? 'marginal';
    const weight = info?.complementWeight ?? 0;
    const intensity = Math.min(1, weight / maxWeight);

    let color: { r: number; g: number; b: number };
    switch (classification) {
      case 'batna':
        // Blue-green: rgb(59, 130, 246) → rgb(34, 197, 94)
        color = {
          r: 0.13 + 0.1 * intensity,
          g: 0.51 + 0.26 * intensity,
          b: 0.96 - 0.59 * intensity,
        };
        break;
      case 'watna':
        // Red-orange: rgb(239, 68, 68) → rgb(249, 115, 22)
        color = {
          r: 0.94 + 0.04 * intensity,
          g: 0.27 + 0.18 * intensity,
          b: 0.27 - 0.18 * intensity,
        };
        break;
      default:
        // Gray
        color = { r: 0.4, g: 0.4, b: 0.4 };
    }

    return { tombstone: t, classification, intensity, color };
  });
}

// ---------------------------------------------------------------------------
// Reservation price surface
// ---------------------------------------------------------------------------

export interface ReservationSurface {
  /** Points on the boundary between BATNA and WATNA regions */
  boundaryPoints: readonly Vec3[];
  /** The reservation "price" in void terms: complement weight at the boundary */
  reservationWeight: number;
  /** Normal direction: which way the BATNA region is */
  batnaDirection: Vec3;
}

/**
 * Compute the reservation price surface: the 3D boundary between
 * BATNA and WATNA tombstone clusters. Points on this surface represent
 * the walk-away threshold -- alternatives at the boundary between
 * acceptable and unacceptable.
 */
export function computeReservationSurface(
  map: VoidMap,
  options?: { eta?: number }
): ReservationSurface {
  const partition = partitionBatnaWatna(map, options);

  if (partition.batna.length === 0 || partition.watna.length === 0) {
    return {
      boundaryPoints: [],
      reservationWeight: partition.reservationWeight,
      batnaDirection: { x: 0, y: 1, z: 0 },
    };
  }

  // Compute centroids of BATNA and WATNA regions
  let batnaCentroid: Vec3 = { x: 0, y: 0, z: 0 };
  for (const alt of partition.batna) {
    batnaCentroid = addVec3(batnaCentroid, alt.centroid);
  }
  batnaCentroid = scaleVec3(batnaCentroid, 1 / partition.batna.length);

  let watnaCentroid: Vec3 = { x: 0, y: 0, z: 0 };
  for (const alt of partition.watna) {
    watnaCentroid = addVec3(watnaCentroid, alt.centroid);
  }
  watnaCentroid = scaleVec3(watnaCentroid, 1 / partition.watna.length);

  // Direction from WATNA to BATNA
  const batnaDir = subtractVec3(batnaCentroid, watnaCentroid);
  const dirLen = lengthVec3(batnaDir);
  const batnaDirection = dirLen > 0 ? scaleVec3(batnaDir, 1 / dirLen) : { x: 0, y: 1, z: 0 };

  // Midpoint is the reservation surface center
  const midpoint = scaleVec3(addVec3(batnaCentroid, watnaCentroid), 0.5);

  // Boundary points: marginal alternatives' centroids, plus interpolated points
  const boundaryPoints: Vec3[] = [];
  for (const alt of partition.marginal) {
    boundaryPoints.push(alt.centroid);
  }

  // If no marginal points, create a grid on the perpendicular plane at midpoint
  if (boundaryPoints.length === 0) {
    // Find two perpendicular vectors
    const up = Math.abs(batnaDirection.y) < 0.9
      ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 };
    const right = {
      x: batnaDirection.y * up.z - batnaDirection.z * up.y,
      y: batnaDirection.z * up.x - batnaDirection.x * up.z,
      z: batnaDirection.x * up.y - batnaDirection.y * up.x,
    };
    const rLen = Math.sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
    const rNorm = rLen > 0 ? scaleVec3(right, 1 / rLen) : { x: 1, y: 0, z: 0 };
    const upPerp = {
      x: batnaDirection.y * rNorm.z - batnaDirection.z * rNorm.y,
      y: batnaDirection.z * rNorm.x - batnaDirection.x * rNorm.z,
      z: batnaDirection.x * rNorm.y - batnaDirection.y * rNorm.x,
    };

    const radius = dirLen * 0.3;
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2;
      boundaryPoints.push(addVec3(midpoint, addVec3(
        scaleVec3(rNorm, Math.cos(angle) * radius),
        scaleVec3(upPerp, Math.sin(angle) * radius)
      )));
    }
  }

  return {
    boundaryPoints,
    reservationWeight: partition.reservationWeight,
    batnaDirection,
  };
}
