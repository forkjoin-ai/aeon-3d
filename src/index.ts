export type TopologyCollapseStrategy = 'race' | 'fold';

export interface TopologyBranch {
  id: string;
  vertices: number;
  drawCalls: number;
  estimatedCostMs: number;
  quality?: number;
  vent?: boolean;
}

export interface TopologyFrameInput {
  branches: TopologyBranch[];
  strategy?: TopologyCollapseStrategy;
  budgetMs?: number;
}

export interface TopologyFrameResult {
  collapsedBy: TopologyCollapseStrategy;
  winnerId: string | null;
  survivingBranches: TopologyBranch[];
  ventedBranchIds: string[];
  totalVertices: number;
  totalDrawCalls: number;
  totalCostMs: number;
  budgetMs: number;
  overBudget: boolean;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export function addVec3(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

export function subtractVec3(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function scaleVec3(value: Vec3, scalar: number): Vec3 {
  return { x: value.x * scalar, y: value.y * scalar, z: value.z * scalar };
}

export function dotVec3(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function lengthVec3(value: Vec3): number {
  return Math.sqrt(dotVec3(value, value));
}

export function normalizeVec3(value: Vec3): Vec3 {
  const length = lengthVec3(value);
  if (length === 0) {
    return { x: 0, y: 0, z: 0 };
  }

  return scaleVec3(value, 1 / length);
}

function ventBranches(branches: TopologyBranch[], budgetMs: number): {
  active: TopologyBranch[];
  ventedIds: string[];
} {
  const ventedIds: string[] = [];
  const explicitActive: TopologyBranch[] = [];

  for (const branch of branches) {
    if (branch.vent) {
      ventedIds.push(branch.id);
      continue;
    }
    explicitActive.push(branch);
  }

  const sortedByCost = [...explicitActive].sort(
    (a, b) => a.estimatedCostMs - b.estimatedCostMs
  );

  const active: TopologyBranch[] = [];
  let runningCost = 0;
  for (const branch of sortedByCost) {
    const wouldFit = runningCost + branch.estimatedCostMs <= budgetMs;
    if (!wouldFit && active.length > 0) {
      ventedIds.push(branch.id);
      continue;
    }

    active.push(branch);
    runningCost += branch.estimatedCostMs;
  }

  return { active, ventedIds };
}

export async function executeTopologyFrame(
  input: TopologyFrameInput
): Promise<TopologyFrameResult> {
  const budgetMs = input.budgetMs ?? 8;
  const strategy = input.strategy ?? 'fold';

  const preparedBranches = input.branches.map((branch) => ({
    ...branch,
    vertices: Math.max(0, Math.round(branch.vertices)),
    drawCalls: Math.max(0, Math.round(branch.drawCalls)),
    estimatedCostMs: Math.max(0, branch.estimatedCostMs),
    quality: branch.quality ?? 1
  }));

  const { active, ventedIds } = ventBranches(preparedBranches, budgetMs);
  const survivingBranches = active.length > 0 ? active : preparedBranches.slice(0, 1);

  let collapsedBy: TopologyCollapseStrategy = strategy;
  let winnerId: string | null = null;
  let collapsedBranches: TopologyBranch[] = survivingBranches;

  if (strategy === 'race') {
    const winner = [...survivingBranches].sort((a, b) => {
      const qualityA = a.quality ?? 1;
      const qualityB = b.quality ?? 1;
      const scoreA = a.estimatedCostMs / qualityA;
      const scoreB = b.estimatedCostMs / qualityB;
      return scoreA - scoreB;
    })[0];

    if (winner) {
      winnerId = winner.id;
      collapsedBranches = [winner];
    } else {
      collapsedBy = 'fold';
    }
  }

  const totals = collapsedBranches.reduce(
    (accumulator, branch) => {
      accumulator.vertices += branch.vertices;
      accumulator.drawCalls += branch.drawCalls;
      accumulator.costMs += branch.estimatedCostMs;
      return accumulator;
    },
    { vertices: 0, drawCalls: 0, costMs: 0 }
  );

  return {
    collapsedBy,
    winnerId,
    survivingBranches: collapsedBranches,
    ventedBranchIds: Array.from(new Set(ventedIds)),
    totalVertices: totals.vertices,
    totalDrawCalls: totals.drawCalls,
    totalCostMs: Number(totals.costMs.toFixed(3)),
    budgetMs,
    overBudget: totals.costMs > budgetMs
  };
}
