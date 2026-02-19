import './TransactionTable.css';

interface Props {
  transactions: any[];
}

export function TransactionTable({ transactions }: Props) {
  if (transactions.length === 0) {
    return <div className="txn-view__empty">No EC transaction data available</div>;
  }

  return (
    <div className="txn-view">
      <div className="txn-table-wrap">
        <table className="txn-table">
          <thead>
            <tr>
              <th>#</th><th>Date</th><th>Doc No</th><th>Type</th>
              <th>From</th><th>To</th><th>Extent</th>
              <th>Survey</th><th>Amount</th><th>SRO</th>
            </tr>
          </thead>
          <tbody>
            {transactions.map((txn: any, i: number) => (
              <tr key={i}>
                <td>{txn.row_number || i + 1}</td>
                <td className="txn-cell--date">{txn.date || '-'}</td>
                <td className="txn-cell--doc">{txn.document_number || '-'}</td>
                <td>
                  <span className={`txn-type txn-type--${(txn.transaction_type || 'other').toLowerCase()}`}>
                    {txn.transaction_type || '-'}
                  </span>
                </td>
                <td>{txn.seller_or_executant || '-'}</td>
                <td>{txn.buyer_or_claimant || '-'}</td>
                <td>{txn.extent || '-'}</td>
                <td>{txn.survey_number || '-'}</td>
                <td className="txn-cell--amount">
                  {txn.consideration_amount ? `₹${Number(txn.consideration_amount).toLocaleString('en-IN')}` : '-'}
                </td>
                <td>{txn.sro || '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
