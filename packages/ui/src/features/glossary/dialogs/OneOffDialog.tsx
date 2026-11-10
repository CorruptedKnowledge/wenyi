import { useEffect, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../../../api/client";
import { useI18n } from "../../../i18n";
import { errorMessage } from "../../../lib/error-codes";
import { toast } from "../../../lib/toast";
import { Button } from "../../../components/ui/button";
import { Dialog } from "../../../components/ui/misc";
import { termTypes } from "../termTypes";

const PREVIEW_ROWS = 20;

/** Find terms that occur fewer than twice in the book, then delete exactly those. */
export function OneOffDialog({
  pid,
  open,
  onClose,
  onSaved,
}: {
  pid: string;
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
}) {
  const { t: tr } = useI18n();
  const find = useMutation({ mutationFn: () => api.oneOffTerms(pid) });
  const remove = useMutation({
    mutationFn: (sources: string[]) => api.bulkDeleteTerms(pid, sources),
    onSuccess: (result) => {
      toast.success(tr("glossary.termsDeleted", { count: result.deleted }));
      onSaved();
    },
    onError: (e) => toast.error(errorMessage(e, tr)),
  });
  // Check again on every opening; the glossary may have changed since the last one.
  const { mutate: check, reset } = find;
  const opened = useRef(false);
  useEffect(() => {
    if (open && !opened.current) check();
    if (!open) reset();
    opened.current = open;
  }, [open, check, reset]);

  const terms = find.data;
  const types = termTypes();
  const working = find.isPending || remove.isPending;
  return (
    <Dialog
      open={open}
      onClose={onClose}
      closeDisabled={remove.isPending}
      title={tr("glossary.oneOffTitle")}
      footer={
        <>
          <Button variant="outline" disabled={remove.isPending} onClick={onClose}>
            {tr("common.cancel")}
          </Button>
          <Button
            variant="destructive"
            disabled={working || !terms || terms.length === 0}
            onClick={() => terms && remove.mutate(terms.map((term) => term.source))}
          >
            {tr("glossary.deleteTheseTerms", { count: terms?.length ?? 0 })}
          </Button>
        </>
      }
    >
      <div className="space-y-3 text-sm">
        {find.isPending && (
          <p role="status" className="text-muted-foreground">
            {tr("glossary.oneOffChecking")}
          </p>
        )}
        {find.isError && <p className="text-destructive">{errorMessage(find.error, tr)}</p>}
        {terms && terms.length === 0 && <p>{tr("glossary.oneOffNone")}</p>}
        {terms && terms.length > 0 && (
          <>
            <p>{tr("glossary.oneOffSummary", { count: terms.length })}</p>
            <p className="text-muted-foreground">{tr("glossary.oneOffBackupHint")}</p>
            <ul className="max-h-64 space-y-1 overflow-y-auto rounded-md border p-2">
              {terms.slice(0, PREVIEW_ROWS).map((term) => (
                <li key={term.source} className="flex gap-2">
                  <span className="font-medium">{term.source}</span>
                  <span className="text-muted-foreground">→ {term.target}</span>
                  <span className="ml-auto text-xs text-muted-foreground">
                    {types[term.type] ?? term.type}
                  </span>
                </li>
              ))}
            </ul>
            {terms.length > PREVIEW_ROWS && (
              <p className="text-muted-foreground">
                {tr("glossary.moreNotShown", { count: terms.length - PREVIEW_ROWS })}
              </p>
            )}
          </>
        )}
      </div>
    </Dialog>
  );
}
