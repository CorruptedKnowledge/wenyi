import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { Conflict } from "../../api/types";
import { useI18n } from "../../i18n";
import { toast } from "../../lib/toast";
import { errorMessage } from "../../lib/error-codes";
import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { PAGE_SIZE } from "./termTypes";
import { TermPagination } from "./TermTable";

/** Unresolved terminology conflicts raised by extraction, resolved inline or all at once. */
export function ConflictList({
  pid,
  conflicts,
  busy,
  onResolved,
}: {
  pid: string;
  conflicts: Conflict[];
  busy: boolean;
  onResolved: () => void;
}) {
  const { t: tr } = useI18n();
  const [page, setPage] = useState(0);
  const lastPage = Math.max(0, Math.ceil(conflicts.length / PAGE_SIZE) - 1);
  const shown = conflicts.slice(
    Math.min(page, lastPage) * PAGE_SIZE,
    (Math.min(page, lastPage) + 1) * PAGE_SIZE,
  );
  const resolve = useMutation({
    mutationFn: ({
      cid,
      decision,
      target,
    }: {
      cid: number;
      decision: string;
      target?: string;
    }) => api.resolveConflict(pid, cid, { decision, target }),
    onSuccess: () => {
      onResolved();
      toast.success(tr("glossary.conflictResolved"));
    },
    onError: (e) => toast.error(errorMessage(e, tr)),
  });
  const keepAll = useMutation({
    mutationFn: () => api.keepCurrentConflicts(pid),
    onSuccess: (result) => {
      onResolved();
      toast.success(tr("glossary.keptCurrentForAll", { count: result.terms }));
    },
    onError: (e) => toast.error(errorMessage(e, tr)),
  });
  const terms = new Set(conflicts.map((c) => c.source)).size;
  return (
    <Card className="border-amber-300 bg-amber-50 dark:bg-amber-950/20">
      <CardContent className="p-4 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <div className="font-medium text-sm">
            {tr("glossary.conflictCount", { count: conflicts.length })}
          </div>
          <Button
            size="sm"
            variant="outline"
            className="ml-auto"
            disabled={busy || keepAll.isPending}
            onClick={() => {
              if (confirm(tr("glossary.keepCurrentForAllConfirm", { count: terms }))) {
                keepAll.mutate();
              }
            }}
          >
            {tr("glossary.keepCurrentForAll")}
          </Button>
          {conflicts.length > PAGE_SIZE && (
            <TermPagination
              page={Math.min(page, lastPage)}
              total={conflicts.length}
              label={tr("glossary.conflictPaginationLabel")}
              onChange={setPage}
            />
          )}
        </div>
        {shown.map((c) => (
          <div
            key={c.id}
            className="flex flex-wrap items-center gap-3 text-sm border-t pt-2"
          >
            <span className="font-medium">{c.source}</span>
            <span className="text-muted-foreground">
              {tr("glossary.current")}
              {c.existing_target}
            </span>
            <span className="text-muted-foreground">
              {tr("glossary.aiSuggestion")}
              {c.proposed_target}
            </span>
            <span className="flex gap-1 ml-auto">
              <Button
                size="sm"
                variant="outline"
                disabled={busy}
                onClick={() => resolve.mutate({ cid: c.id, decision: "current" })}
              >
                {tr("glossary.keepCurrent")}
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={busy}
                onClick={() => resolve.mutate({ cid: c.id, decision: "proposed" })}
              >
                {tr("glossary.acceptSuggestion")}
              </Button>
            </span>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
