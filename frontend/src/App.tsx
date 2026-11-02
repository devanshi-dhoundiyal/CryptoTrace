
import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import TransactionGraph, { type Transaction } from './TransactionGraph'
import {
  Activity,
  Blocks,
  FileSearch,
  LayoutDashboard,
  Network,
  Search,
  ShieldAlert,
  ShieldCheck,
  RefreshCw,
} from 'lucide-react'

const API = 'http://127.0.0.1:8000'

type Summary = {
  dataset_transactions: number
  labelled_prediction_rows: number
  graph_connections: number
  predicted_illicit: number
  predicted_licit: number
  prediction_coverage: string
}

type GraphData = {
  nodes: Transaction[]
  edges: { id: string; source: string; target: string }[]
  seed_transactions: string[]
  note: string
}

type TransactionDetails = Transaction & {
  neighbors?: Transaction[]
  neighbor_count?: number
}

function formatNumber(value: number) {
  return value.toLocaleString('en-IN')
}

function getRiskStyle(risk: Transaction['risk']) {
  switch (risk) {
    case 'High':
      return 'bg-red-400/10 text-red-300'
    case 'Medium':
      return 'bg-amber-400/10 text-amber-300'
    case 'Low':
      return 'bg-emerald-400/10 text-emerald-300'
    default:
      return 'bg-slate-400/10 text-slate-300'
  }
}

function App() {
  const [activePage, setActivePage] = useState('Overview')
  const [summary, setSummary] = useState<Summary | null>(null)
  const [transactions, setTransactions] = useState<Transaction[]>([])

  const [graph, setGraph] = useState<GraphData>({
    nodes: [],
    edges: [],
    seed_transactions: [],
    note: '',
  })

  const [selectedTransaction, setSelectedTransaction] =
    useState<TransactionDetails | null>(null)

  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [searchError, setSearchError] = useState('')
  const [graphLoading, setGraphLoading] = useState(false)
  const [investigating, setInvestigating] = useState(false)

  // Load dashboard summary
  const loadSummary = useCallback(async () => {
    const response = await fetch(`${API}/api/summary`)

    if (!response.ok) {
      throw new Error('Could not load dashboard summary.')
    }

    const data: Summary = await response.json()
    setSummary(data)
  }, [])

  // Load transaction list
  const loadTransactions = useCallback(async (search = '') => {
    const params = new URLSearchParams({
      limit: '50',
      search,
    })

    const response = await fetch(`${API}/api/transactions?${params}`)

    if (!response.ok) {
      throw new Error('Could not load transactions.')
    }

    const data = await response.json()
    setTransactions(data.transactions)
  }, [])

  // Load default transaction graph
  const loadGraph = useCallback(async () => {
    setGraphLoading(true)

    try {
      const response = await fetch(`${API}/api/graph?limit=40`)

      if (!response.ok) {
        throw new Error('Could not load transaction graph.')
      }

      const data: GraphData = await response.json()
      setGraph(data)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Graph loading failed.',
      )
    } finally {
      setGraphLoading(false)
    }
  }, [])

  // Load initial application data
  const loadDashboard = useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      await Promise.all([
        loadSummary(),
        loadTransactions(),
        loadGraph(),
      ])
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to connect to the backend.',
      )
    } finally {
      setLoading(false)
    }
  }, [loadSummary, loadTransactions, loadGraph])

  useEffect(() => {
    loadDashboard()
  }, [loadDashboard])

  // Investigate an exact transaction ID
  const handleSearch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSearchError('')

    const txId = searchQuery.trim()

    if (!/^\d+$/.test(txId)) {
      setSearchError('Please enter a valid numeric transaction ID.')
      return
    }

    setInvestigating(true)

    try {
      // Fetch exact transaction, including unknown predictions
      const transactionResponse = await fetch(
        `${API}/api/transactions/${txId}`,
      )

      if (!transactionResponse.ok) {
        if (transactionResponse.status === 404) {
          throw new Error(
            'Transaction not found in the Elliptic dataset.',
          )
        }

        throw new Error('Could not retrieve transaction details.')
      }

      const transaction: TransactionDetails =
        await transactionResponse.json()

      // Fetch its actual recorded neighbouring transactions
      const graphResponse = await fetch(
        `${API}/api/graph?seed=${txId}&limit=40`,
      )

      if (!graphResponse.ok) {
        throw new Error('Could not load connected transactions.')
      }

      const graphData: GraphData = await graphResponse.json()

      setSelectedTransaction(transaction)
      setGraph(graphData)
      setActivePage('Transaction Network')
    } catch (err) {
      setSearchError(
        err instanceof Error
          ? err.message
          : 'Unable to investigate this transaction.',
      )
    } finally {
      setInvestigating(false)
    }
  }

  // Inspect a transaction selected from a table or graph
  const handleSelectTransaction = async (transaction: Transaction) => {
    setSelectedTransaction(transaction)
    setSearchError('')

    try {
      const response = await fetch(
        `${API}/api/transactions/${transaction.id}`,
      )

      if (!response.ok) {
        throw new Error('Could not load transaction details.')
      }

      const data: TransactionDetails = await response.json()
      setSelectedTransaction(data)
    } catch (err) {
      setSearchError(
        err instanceof Error
          ? err.message
          : 'Unable to load transaction.',
      )
    }
  }

  // Search from investigation page and navigate to network
  const navItems = [
    { label: 'Overview', icon: LayoutDashboard },
    { label: 'Transaction Investigation', icon: FileSearch },
    { label: 'Transaction Network', icon: Network },
  ]

  const predictedPercentage =
    summary && summary.labelled_prediction_rows > 0
      ? (
          (summary.predicted_illicit /
            summary.labelled_prediction_rows) *
          100
        ).toFixed(2)
      : '0.00'

  const networkRiskCounts = {
  High: graph.nodes.filter((tx) => tx.risk === 'High').length,
  Medium: graph.nodes.filter((tx) => tx.risk === 'Medium').length,
  Low: graph.nodes.filter((tx) => tx.risk === 'Low').length,
  Unknown: graph.nodes.filter((tx) => tx.risk === 'Unknown').length,
}

  const showTransactionDetails = (
    transaction: TransactionDetails,
  ) => (
    <div className="rounded-2xl border border-white/10 bg-[#0D1423] p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h4 className="font-semibold text-white">
          Transaction details
        </h4>

        <span
          className={`rounded-full px-3 py-1 text-xs ${getRiskStyle(
            transaction.risk,
          )}`}
        >
          {transaction.risk} risk
        </span>
      </div>

      <p className="mt-4 break-all font-mono text-sm text-cyan-200">
        {transaction.id}
      </p>

      <div className="mt-5 grid gap-4 sm:grid-cols-3">
        <div>
          <p className="text-xs text-slate-500">Prediction</p>
          <p className="mt-1 text-sm text-white">
            {!transaction.has_prediction
              ? 'Unknown'
              : transaction.predicted_class === 1
                ? 'Illicit'
                : 'Licit'}
          </p>
        </div>

        <div>
          <p className="text-xs text-slate-500">
            Illicit probability
          </p>
          <p className="mt-1 text-sm text-white">
            {transaction.illicit_probability !== null
              ? `${(
                  transaction.illicit_probability * 100
                ).toFixed(2)}%`
              : 'Not available'}
          </p>
        </div>

        <div>
          <p className="text-xs text-slate-500">Time step</p>
          <p className="mt-1 text-sm text-white">
            {transaction.time_step ?? 'Not available'}
          </p>
        </div>
      </div>

      {transaction.neighbors && (
        <div className="mt-5 border-t border-white/10 pt-4">
          <p className="text-sm text-slate-300">
            Connected transactions:{' '}
            <span className="font-semibold text-white">
              {transaction.neighbor_count ?? 0}
            </span>
          </p>
          <p className="mt-2 text-xs text-slate-500">
            Showing up to {transaction.neighbors.length} neighbours
            in the transaction details response.
          </p>
        </div>
      )}
    </div>
  )

  return (
    <div className="min-h-screen bg-[#080D19] text-slate-200">

      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-64 flex-col border-r border-white/[0.07] bg-[#0B1120] lg:flex">
        <div className="flex h-20 items-center gap-3 border-b border-white/[0.07] px-6">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-400/10 text-cyan-300">
            <ShieldCheck size={23} />
          </div>

          <div>
            <h1 className="text-lg font-bold text-white">
              CryptoTrace
            </h1>
            <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">
              Research Prototype
            </p>
          </div>
        </div>

        <div className="px-4 pt-7">
          <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
            Workspace
          </p>

          <nav className="space-y-1">
            {navItems.map(({ label, icon: Icon }) => (
              <button
                key={label}
                onClick={() => setActivePage(label)}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm transition ${
                  activePage === label
                    ? 'bg-cyan-400/10 font-medium text-cyan-300'
                    : 'text-slate-400 hover:bg-white/[0.04] hover:text-white'
                }`}
              >
                <Icon size={18} />
                {label}
              </button>
            ))}
          </nav>
        </div>

        <div className="mt-auto p-4">
          <div className="rounded-2xl border border-white/[0.08] bg-white/[0.025] p-4">
            <div className="mb-3 flex items-center gap-2 text-cyan-300">
              <Activity size={17} />
              <span className="text-xs font-semibold">
                Model status
              </span>
            </div>

            <p className="text-sm font-medium text-white">
              XGBoost
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Local research model
            </p>

            <div className="mt-4 flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${
                  error ? 'bg-red-400' : 'bg-emerald-400'
                }`}
              />
              <span className="text-xs text-slate-400">
                {error ? 'Connection issue' : 'Backend connected'}
              </span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="min-h-screen lg:ml-64">

        <header className="sticky top-0 z-10 flex h-20 items-center justify-between border-b border-white/[0.07] bg-[#080D19]/90 px-5 backdrop-blur-xl sm:px-8">
          <div>
            <p className="text-xs text-slate-500">
              CryptoTrace / {activePage}
            </p>
            <h2 className="mt-1 text-lg font-semibold text-white">
              {activePage}
            </h2>
          </div>

          <button
            onClick={loadDashboard}
            disabled={loading}
            className="flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-300 hover:bg-white/5 disabled:opacity-50"
          >
            <RefreshCw
              size={15}
              className={loading ? 'animate-spin' : ''}
            />
            Refresh
          </button>
        </header>

        <div className="mx-auto max-w-[1500px] space-y-7 p-5 sm:p-8">

          {error && (
            <div className="rounded-xl border border-red-400/20 bg-red-400/10 p-4 text-sm text-red-300">
              {error}
            </div>
          )}

          {loading && !summary ? (
            <div className="rounded-2xl border border-white/10 bg-[#0D1423] p-10 text-center text-slate-400">
              Loading CryptoTrace data...
            </div>
          ) : (
            <>

              {/* OVERVIEW */}
              {activePage === 'Overview' && (
                <>
                  <section>
                    <div className="mb-2 flex items-center gap-2 text-xs font-medium text-cyan-300">
                      <span className="h-1.5 w-1.5 rounded-full bg-cyan-300" />
                      CRYPTOCURRENCY TRANSACTION INTELLIGENCE
                    </div>

                    <h3 className="text-2xl font-semibold text-white sm:text-3xl">
                      Investigation overview
                    </h3>

                    <p className="mt-2 max-w-xl text-sm leading-6 text-slate-400">
                      Explore transaction predictions and recorded
                      relationships using the XGBoost research model.
                    </p>
                  </section>

                  <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                    {[
                      {
                        title: 'Dataset transactions',
                        value: summary
                          ? formatNumber(summary.dataset_transactions)
                          : '—',
                        icon: Blocks,
                        color: 'text-cyan-300',
                      },
                      {
                        title: 'Transactions analysed',
                        value: summary
                          ? formatNumber(summary.labelled_prediction_rows)
                          : '—',
                        icon: FileSearch,
                        color: 'text-violet-300',
                      },
                      {
                        title: 'Predicted illicit',
                        value: summary
                          ? formatNumber(summary.predicted_illicit)
                          : '—',
                        icon: ShieldAlert,
                        color: 'text-red-300',
                      },
                      {
                        title: 'Predicted licit',
                        value: summary
                          ? formatNumber(summary.predicted_licit)
                          : '—',
                        icon: ShieldCheck,
                        color: 'text-emerald-300',
                      },
                    ].map((metric) => (
                      <div
                        key={metric.title}
                        className="rounded-2xl border border-white/[0.08] bg-[#0D1423] p-5"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-slate-400">
                            {metric.title}
                          </span>
                          <metric.icon
                            size={19}
                            className={metric.color}
                          />
                        </div>

                        <p className="mt-5 text-2xl font-semibold text-white">
                          {metric.value}
                        </p>
                      </div>
                    ))}
                  </section>

                  <section className="grid gap-4 lg:grid-cols-2">
                    <div className="rounded-2xl border border-white/10 bg-[#0D1423] p-5">
                      <p className="text-sm text-slate-400">
                        Illicit prediction share
                      </p>
                      <p className="mt-3 text-3xl font-semibold text-white">
                        {predictedPercentage}%
                      </p>
                      <p className="mt-2 text-xs text-slate-500">
                        Of labelled test transactions predicted by XGBoost
                      </p>
                    </div>

                    <div className="rounded-2xl border border-white/10 bg-[#0D1423] p-5">
                      <p className="text-sm text-slate-400">
                        Recorded graph connections
                      </p>
                      <p className="mt-3 text-3xl font-semibold text-white">
                        {summary
                          ? formatNumber(summary.graph_connections)
                          : '—'}
                      </p>
                      <p className="mt-2 text-xs text-slate-500">
                        Transaction relationships in the Elliptic dataset
                      </p>
                    </div>
                  </section>

                  <section className="rounded-2xl border border-white/10 bg-[#0D1423] p-5">
                    <h4 className="font-semibold text-white">
                      Prediction coverage
                    </h4>
                    <p className="mt-2 text-sm text-slate-400">
                      {summary?.prediction_coverage}
                    </p>
                    <p className="mt-2 text-xs leading-5 text-slate-500">
                      Only transactions with saved XGBoost predictions
                      have a prediction score. Other dataset transactions
                      remain unscored.
                    </p>
                  </section>
                </>
              )}

              {/* TRANSACTION INVESTIGATION */}
              {activePage === 'Transaction Investigation' && (
                <>
                  <section>
                    <h3 className="text-2xl font-semibold text-white">
                      Transaction investigation
                    </h3>
                    <p className="mt-2 text-sm text-slate-400">
                      Enter a transaction ID to investigate it and
                      explore its recorded network connections.
                    </p>
                  </section>

                  <form
                    onSubmit={handleSearch}
                    className="flex flex-col gap-3 sm:flex-row"
                  >
                    <div className="relative flex-1">
                      <Search
                        size={17}
                        className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                      />

                      <input
                        value={searchQuery}
                        onChange={(event) =>
                          setSearchQuery(event.target.value)
                        }
                        placeholder="Enter numeric transaction ID..."
                        className="w-full rounded-xl border border-white/10 bg-[#0D1423] py-3 pl-10 pr-4 text-sm text-white outline-none focus:border-cyan-300/50"
                      />
                    </div>

                    <button
                      type="submit"
                      disabled={investigating}
                      className="rounded-xl bg-cyan-300 px-6 py-3 text-sm font-semibold text-slate-950 hover:bg-cyan-200 disabled:opacity-50"
                    >
                      {investigating ? 'Investigating...' : 'Investigate'}
                    </button>
                  </form>

                  {searchError && (
                    <div className="rounded-xl border border-red-400/20 bg-red-400/10 p-4 text-sm text-red-300">
                      {searchError}
                    </div>
                  )}

                  {selectedTransaction && (
                    <div className="space-y-4">
                      {showTransactionDetails(selectedTransaction)}

                      <button
                        onClick={() => setActivePage('Transaction Network')}
                        className="flex items-center gap-2 rounded-xl bg-cyan-300 px-5 py-3 text-sm font-semibold text-slate-950 hover:bg-cyan-200"
                      >
                        <Network size={16} />
                        View connected transactions
                      </button>
                    </div>
                  )}

                  <div className="overflow-hidden rounded-2xl border border-white/10 bg-[#0D1423]">
                    <div className="border-b border-white/10 p-5">
                      <h4 className="font-semibold text-white">
                        Available predicted transactions
                      </h4>
                      <p className="mt-1 text-xs text-slate-500">
                        Select a transaction to inspect it.
                      </p>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full min-w-[650px] text-left text-sm">
                        <thead className="bg-white/[0.02] text-xs text-slate-500">
                          <tr>
                            <th className="px-5 py-3">Transaction ID</th>
                            <th className="px-5 py-3">Time step</th>
                            <th className="px-5 py-3">Prediction</th>
                            <th className="px-5 py-3">Illicit probability</th>
                            <th className="px-5 py-3">Risk</th>
                          </tr>
                        </thead>

                        <tbody className="divide-y divide-white/5">
                          {transactions.map((tx) => (
                            <tr
                              key={tx.id}
                              onClick={() => handleSelectTransaction(tx)}
                              className="cursor-pointer hover:bg-white/[0.04]"
                            >
                              <td className="px-5 py-4 font-mono text-xs text-cyan-200">
                                {tx.id}
                              </td>

                              <td className="px-5 py-4 text-slate-300">
                                {tx.time_step ?? '—'}
                              </td>

                              <td className="px-5 py-4 text-slate-300">
                                {!tx.has_prediction
                                  ? 'Unknown'
                                  : tx.predicted_class === 1
                                    ? 'Illicit'
                                    : 'Licit'}
                              </td>

                              <td className="px-5 py-4 text-slate-300">
                                {tx.illicit_probability !== null
                                  ? `${(
                                      tx.illicit_probability * 100
                                    ).toFixed(2)}%`
                                  : '—'}
                              </td>

                              <td className="px-5 py-4">
                                <span
                                  className={`rounded-full px-2.5 py-1 text-xs ${getRiskStyle(
                                    tx.risk,
                                  )}`}
                                >
                                  {tx.risk}
                                </span>
                              </td>
                            </tr>
                          ))}

                          {transactions.length === 0 && (
                            <tr>
                              <td
                                colSpan={5}
                                className="px-5 py-8 text-center text-slate-500"
                              >
                                No transactions available.
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}

              {/* TRANSACTION NETWORK */}
              {activePage === 'Transaction Network' && (
                <>
                  <section>
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <h3 className="text-2xl font-semibold text-white">
                          Transaction network
                        </h3>
                        <p className="mt-2 text-sm text-slate-400">
                          Explore recorded transaction relationships
                          and inspect connected transactions.
                        </p>
                      </div>

                      {selectedTransaction && (
  showTransactionDetails(selectedTransaction)
)}

{/* Network Risk Summary */}
<section>
  <div className="mb-4">
    <h4 className="text-lg font-semibold text-white">
      Network risk summary
    </h4>
    <p className="mt-1 text-sm text-slate-400">
      Risk distribution of transactions currently displayed
      in the investigated network.
    </p>
  </div>

  <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
    {[
      {
        label: 'High risk',
        value: networkRiskCounts.High,
        color: 'text-red-300',
        border: 'border-red-400/20',
        background: 'bg-red-400/5',
      },
      {
        label: 'Medium risk',
        value: networkRiskCounts.Medium,
        color: 'text-amber-300',
        border: 'border-amber-400/20',
        background: 'bg-amber-400/5',
      },
      {
        label: 'Low risk',
        value: networkRiskCounts.Low,
        color: 'text-emerald-300',
        border: 'border-emerald-400/20',
        background: 'bg-emerald-400/5',
      },
      {
        label: 'Unknown',
        value: networkRiskCounts.Unknown,
        color: 'text-slate-300',
        border: 'border-slate-400/20',
        background: 'bg-slate-400/5',
      },
    ].map((item) => (
      <div
        key={item.label}
        className={`rounded-xl border ${item.border} ${item.background} p-4`}
      >
        <p className="text-sm text-slate-400">
          {item.label}
        </p>
        <p className={`mt-2 text-3xl font-semibold ${item.color}`}>
          {item.value}
        </p>
      </div>
    ))}
  </div>
</section>

<div className="overflow-hidden rounded-2xl border border-white/10 bg-[#0D1423]"></div>
                    </div>
                  </section>

                  {selectedTransaction && (
                    showTransactionDetails(selectedTransaction)
                  )}

                  <div className="overflow-hidden rounded-2xl border border-white/10 bg-[#0D1423]">
                    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 p-5">
                      <div>
                        <h4 className="font-semibold text-white">
                          Elliptic transaction graph
                        </h4>
                        <p className="mt-1 text-xs text-slate-500">
                          {graph.nodes.length} nodes · {graph.edges.length}{' '}
                          connections displayed
                        </p>
                      </div>

                      {graphLoading && (
                        <span className="text-xs text-cyan-300">
                          Loading graph...
                        </span>
                      )}
                    </div>

                    <TransactionGraph
                      transactions={graph.nodes}
                      edges={graph.edges}
                      onSelect={handleSelectTransaction}
                    />

                    <div className="flex flex-wrap gap-4 border-t border-white/10 px-5 py-3 text-xs text-slate-400">
                      {(
                        ['High', 'Medium', 'Low', 'Unknown'] as const
                      ).map((risk) => (
                        <span
                          key={risk}
                          className="flex items-center gap-2"
                        >
                          <span
                            className="h-2 w-2 rounded-full"
                            style={{
                              backgroundColor:
                                risk === 'High'
                                  ? '#F87171'
                                  : risk === 'Medium'
                                    ? '#FBBF24'
                                    : risk === 'Low'
                                      ? '#34D399'
                                      : '#94A3B8',
                            }}
                          />
                          {risk}
                        </span>
                      ))}
                    </div>
                  </div>

                  {searchError && (
                    <p className="text-sm text-red-300">{searchError}</p>
                  )}
                </>
              )}

            </>
          )}

          <footer className="border-t border-white/5 pt-5 text-xs text-slate-600">
            CryptoTrace · Academic research prototype · XGBoost predictions
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
