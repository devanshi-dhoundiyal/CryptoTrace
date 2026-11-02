
import { useCallback, useMemo } from 'react'
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  Handle,
  Position,
  type Node,
  type Edge,
  type NodeProps,
  type NodeTypes,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

export type Transaction = {
  id: string
  time_step: number | null
  has_prediction: boolean
  predicted_class: number | null
  illicit_probability: number | null
  risk: 'High' | 'Medium' | 'Low' | 'Unknown'
}

type TransactionNodeData = {
  transaction: Transaction
  onSelect: (transaction: Transaction) => void
}

type GraphEdge = {
  id: string
  source: string
  target: string
}

type Props = {
  transactions: Transaction[]
  edges: GraphEdge[]
  onSelect: (transaction: Transaction) => void
}

function getRiskColor(risk: Transaction['risk']) {
  return {
    High: '#F87171',
    Medium: '#FBBF24',
    Low: '#34D399',
    Unknown: '#94A3B8',
  }[risk]
}

function TransactionNode({ data }: NodeProps<Node<TransactionNodeData>>) {
  const { transaction, onSelect } = data
  const color = getRiskColor(transaction.risk)

  return (
    <div
      onClick={() => onSelect(transaction)}
      title={`Inspect transaction ${transaction.id}`}
      className="min-w-[145px] cursor-pointer rounded-xl border border-white/10 bg-[#111B2D] px-3 py-3 shadow-xl transition hover:border-cyan-300/60"
      style={{ boxShadow: `0 0 22px ${color}18` }}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!border-2 !border-[#111B2D] !bg-slate-400"
      />

      <div className="flex items-center gap-2">
        <span
          className="h-2.5 w-2.5 shrink-0 rounded-full"
          style={{ backgroundColor: color }}
        />
        <span className="font-mono text-xs font-semibold text-white">
          {transaction.id}
        </span>
      </div>

      <div className="mt-2 flex items-center justify-between gap-2">
        <span className="text-[10px] text-slate-400">
          {transaction.risk} risk
        </span>

        <span className="text-[10px] font-semibold" style={{ color }}>
          {transaction.has_prediction &&
          transaction.illicit_probability !== null
            ? `${(transaction.illicit_probability * 100).toFixed(1)}%`
            : 'N/A'}
        </span>
      </div>

      <Handle
        type="source"
        position={Position.Right}
        className="!border-2 !border-[#111B2D] !bg-slate-400"
      />
    </div>
  )
}

const nodeTypes: NodeTypes = {
  transaction: TransactionNode,
}

export default function TransactionGraph({
  transactions,
  edges,
  onSelect,
}: Props) {
  const handleSelect = useCallback(
    (transaction: Transaction) => onSelect(transaction),
    [onSelect],
  )

  const nodes: Node<TransactionNodeData>[] = useMemo(
    () =>
      transactions.map((transaction, index) => ({
        id: transaction.id,
        type: 'transaction',
        position: {
          x: (index % 4) * 230,
          y: Math.floor(index / 4) * 170,
        },
        data: {
          transaction,
          onSelect: handleSelect,
        },
      })),
    [transactions, handleSelect],
  )

  const flowEdges: Edge[] = useMemo(
    () =>
      edges.map((edge) => ({
        ...edge,
        type: 'smoothstep',
        style: {
          stroke: '#52647E',
          strokeWidth: 1.5,
        },
      })),
    [edges],
  )

  return (
    <div className="h-[390px] w-full sm:h-[450px]">
      {transactions.length === 0 ? (
        <div className="flex h-full items-center justify-center text-sm text-slate-400">
          No transaction graph data available.
        </div>
      ) : (
        <ReactFlow
          nodes={nodes}
          edges={flowEdges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.2}
          maxZoom={2}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#334155" gap={22} size={1} />

          <Controls
            position="top-right"
            className="!overflow-hidden !rounded-lg !border !border-white/10 !bg-[#111B2D] [&_button]:!border-white/10 [&_button]:!bg-[#111B2D] [&_button]:!fill-slate-300"
          />

          <MiniMap
            position="bottom-left"
            pannable
            zoomable
            nodeColor={(node) => {
              const data = node.data as TransactionNodeData
              return getRiskColor(data.transaction.risk)
            }}
            maskColor="rgba(8,13,25,0.45)"
            className="!border !border-white/10 !bg-[#111B2D]"
          />
        </ReactFlow>
      )}
    </div>
  )
}
