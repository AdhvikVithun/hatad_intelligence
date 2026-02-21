import { useState, useCallback } from 'react';
import { useAnalysis } from './hooks/useAnalysis';
import { Bootloader } from './components/boot/Bootloader';
import { TopBar } from './components/layout/TopBar';
import { Sidebar } from './components/layout/Sidebar';
import { TabBar } from './components/layout/TabBar';
import { RiskPanel } from './components/layout/RiskPanel';
import { BottomBar } from './components/layout/BottomBar';
import { ToastContainer } from './components/common/Toast';
import { WelcomeScreen } from './components/WelcomeScreen';
import { PipelineView } from './components/pipeline/PipelineView';
import { LogView } from './components/results/LogView';
import { SummaryView } from './components/results/SummaryView';
import { ChecksTable } from './components/results/ChecksTable';
import { ChainView } from './components/results/ChainView';
import { TransactionTable } from './components/results/TransactionTable';
import { KnowledgeBase } from './components/results/KnowledgeBase';
import { NarrativeReport } from './components/results/NarrativeReport';
import { IdentityPanel } from './components/results/IdentityPanel';
import { EmptyState } from './components/common/EmptyState';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import './components/common/ErrorBoundary.css';

export default function App() {
  const [booted, setBooted] = useState(false);
  const handleBootReady = useCallback(() => setBooted(true), []);

  const a = useAnalysis();

  const showWelcome = !a.session && !a.processing && a.progress.length === 0 && !['log'].includes(a.activeTab);

  /* ── Bootloader gate ── */
  if (!booted) {
    return <Bootloader onReady={handleBootReady} />;
  }

  return (
    <div className="app-container">
      <ToastContainer toasts={a.toasts} onDismiss={a.dismissToast} />

      <TopBar
        processing={a.processing}
        session={a.session}
        riskScore={a.riskScore}
        riskBand={a.riskBand}
        riskColor={a.riskColor}
        elapsed={a.elapsed}
        pipeline={a.pipeline}
        llmOnline={a.llmOnline}
        clock={a.clock}
        formatDuration={a.formatDuration}
      />

      <div className="main-content">
        <Sidebar
          files={a.files}
          uploading={a.uploading}
          processing={a.processing}
          session={a.session}
          sessionId={a.sessionId}
          dragging={a.dragging}
          llmOnline={a.llmOnline}
          sessionHistory={a.sessionHistory}
          fileInputRef={a.fileInputRef}
          onUpload={a.handleUpload}
          onDrop={a.handleDrop}
          onStartAnalysis={a.handleStartAnalysis}
          onClear={a.handleClear}
          onLoadHistory={a.loadSessionHistory}
          onLoadSession={a.loadPastSession}
          setDragging={a.setDragging}
          getReportPdfUrl={a.getReportPdfUrl}
        />

        <div className="center-panel">
          <TabBar
            tabs={a.tabs}
            activeTab={a.activeTab}
            processing={a.processing}
            onChange={a.setActiveTab}
          />

          <div className="tab-content">
          <ErrorBoundary>
            {showWelcome && <WelcomeScreen />}

            {/* Failed-state banner: analysis finished but session is null */}
            {a.activeTab === 'pipeline' && !a.processing && !a.session && a.progress.length > 0 && (
              <div className="pipeline-failed-banner">
                <span className="material-icons pipeline-failed-banner__icon">error_outline</span>
                <div className="pipeline-failed-banner__text">
                  <div className="pipeline-failed-banner__title">Analysis did not complete successfully</div>
                  <div className="pipeline-failed-banner__desc">
                    Review the pipeline log above for error details, or clear and start a new analysis.
                  </div>
                </div>
              </div>
            )}

            {a.activeTab === 'pipeline' && (a.processing || a.progress.length > 0) && (
              <PipelineView
                pipeline={a.pipeline}
                progress={a.progress}
                processing={a.processing}
                session={a.session}
                riskScore={a.riskScore}
                riskBand={a.riskBand}
                riskColor={a.riskColor}
                passCount={a.passCount}
                failCount={a.failCount}
                warnCount={a.warnCount}
                elapsed={a.elapsed}
                llmActive={a.llmActive}
                progressEndRef={a.progressEndRef}
                formatDuration={a.formatDuration}
              />
            )}

            {a.activeTab === 'log' && <LogView progress={a.progress} />}

            {a.activeTab === 'summary' && a.session && a.verification && (
              <SummaryView
                verification={a.verification}
                redFlags={a.redFlags}
                criticalChecks={a.criticalChecks}
                recommendations={a.recommendations}
                missingDocs={a.missingDocs}
              />
            )}
            {a.activeTab === 'summary' && !a.session && !a.processing && (
              <EmptyState title="No results" description="Upload documents and run analysis to see results" />
            )}

            {a.activeTab === 'checks' && a.session && <ChecksTable checks={a.checks} />}
            {a.activeTab === 'checks' && !a.session && (
              <EmptyState title="No verification data" description="Run analysis to see check results" />
            )}

            {a.activeTab === 'chain' && a.session && (
              <ChainView chain={a.chain} chainGraphData={a.chainGraphData} />
            )}
            {a.activeTab === 'chain' && !a.session && (
              <EmptyState title="No chain data" description="Run analysis to see chain of title" />
            )}

            {a.activeTab === 'transactions' && a.session && (
              <TransactionTable transactions={a.transactions} />
            )}
            {a.activeTab === 'transactions' && !a.session && (
              <EmptyState title="No transaction data" description="Upload EC documents to see transactions" />
            )}

            {a.activeTab === 'identity' && (a.processing || a.session) && (
              <IdentityPanel clusters={a.identityClusters} />
            )}
            {a.activeTab === 'identity' && !a.session && !a.processing && (
              <EmptyState title="No identity data" description="Run analysis to see identity resolution" />
            )}

            {a.activeTab === 'knowledge' && (a.processing || a.session) && (
              <KnowledgeBase
                processing={a.processing}
                pipeline={a.pipeline}
                memoryBank={a.memoryBank}
                mbFacts={a.mbFacts}
                mbConflicts={a.mbConflicts}
                mbCrossRefs={a.mbCrossRefs}
                mbCategories={a.mbCategories}
              />
            )}
            {a.activeTab === 'knowledge' && !a.session && !a.processing && (
              <EmptyState title="No knowledge base" description="Upload documents and run analysis to build the knowledge base" />
            )}

            {a.activeTab === 'report' && a.session && (
              <NarrativeReport report={a.session.narrative_report} />
            )}
            {a.activeTab === 'report' && !a.session && (
              <EmptyState title="No report" description="Run analysis to generate a narrative report" />
            )}
          </ErrorBoundary>
          </div>
        </div>

        <RiskPanel
          processing={a.processing}
          session={a.session}
          riskScore={a.riskScore}
          riskBand={a.riskBand}
          riskColor={a.riskColor}
          elapsed={a.elapsed}
          files={a.files}
          checks={a.checks}
          passCount={a.passCount}
          failCount={a.failCount}
          warnCount={a.warnCount}
          redFlags={a.redFlags}
          transactions={a.transactions}
          pipeline={a.pipeline}
          formatDuration={a.formatDuration}
        />
      </div>

      <BottomBar
        sessionId={a.sessionId}
        processing={a.processing}
        session={a.session}
        llmOnline={a.llmOnline}
        llmActive={a.llmActive}
        lastLlmEvent={a.lastLlmEvent}
      />
    </div>
  );
}
