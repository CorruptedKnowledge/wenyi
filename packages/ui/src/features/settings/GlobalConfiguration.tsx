import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useIsMutating, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "../../lib/toast";
import { api } from "../../api/client";
import type { GlobalConfig } from "../../api/types";
import { platform } from "../../platform";
import { LazyBoundary } from "../../routes/LazyBoundary";
import type { SettingsSectionId } from "../../routes/manifest";
import { useI18n } from "../../i18n";
import { Button } from "../../components/ui/button";
import { ErrorNotice } from "../../components/ui/data";
import { Textarea } from "../../components/ui/form";
import { LanguageSettings } from "./LanguageSettings";
import { cardSurface } from "../../components/ui/card";
import { cn } from "../../lib/utils";
import { ModelSelection } from "./ModelSelection";
import { ExportSettings } from "./ExportSettings";
import { ProviderSettings } from "./ProviderSettings";
import { PdfSettings } from "./PdfSettings";
import { SettingsActionBar, SettingsGroup, SettingsSection } from "./SettingsSection";
import { WorkflowSettings, workflowField } from "./WorkflowSettings";
import { renameRegistryId, type RegistryGroup } from "./registryEdits";
import { SETTINGS_CONTENT_CLASSNAME } from "./SettingsRow";

type Document = Record<string, unknown>;
const object = (value: unknown) => (value || {}) as Document;
const noPendingCredentials = () => false;

export function GlobalConfiguration({ section }: { section: SettingsSectionId }) {
  const { t } = useI18n();
  const location = useLocation();
  const qc = useQueryClient();
  const credentials = platform().capabilities.credentials;
  // The platform is fixed before React mounts, so the hook identity is stable.
  const useCredentialsPending = credentials?.usePending ?? noPendingCredentials;
  const credentialsPending = useCredentialsPending();
  const Updates = platform().capabilities.updates?.Section;
  const Diagnostics = platform().capabilities.diagnostics?.Section;
  const mutations = useIsMutating();
  const [installing, setInstalling] = useState(false);
  const query = useQuery({
    queryKey: ["globalConfig"],
    queryFn: api.getGlobalConfig,
  });
  const { data: caps } = useQuery({
    queryKey: ["capabilities"],
    queryFn: api.capabilities,
  });
  const [draft, setDraft] = useState<GlobalConfig | null>(null);
  const [yamlDirty, setYamlDirty] = useState(false);
  // A toast disappears; a key that failed to reach the credential store stays
  // unsaved until the user acts, so the reason has to stay on the page.
  const [credentialsFailed, setCredentialsFailed] = useState(false);
  const [uncommitted, setUncommitted] = useState(false);
  const [registryKey, setRegistryKey] = useState(0);
  const [editingIds, setEditingIds] = useState(false);
  const [renames, setRenames] = useState<Record<string, string>>({});
  const [providerRenames, setProviderRenames] = useState<Record<string, string>>({});
  useEffect(() => {
    if (query.data && !draft) setDraft(query.data);
  }, [query.data, draft]);
  useEffect(() => () => credentials?.retain([]), [credentials]);
  const input = () => ({
    yaml: draft!.yaml,
    revision: draft!.revision,
    default_template: "标准翻译" as const,
    model_renames: renames,
    ...(credentials ? { provider_renames: providerRenames } : {}),
  });
  const apply = (value: GlobalConfig) => {
    setDraft(value);
    setYamlDirty(false);
  };
  // Fields stay editable while a request runs. A reply answers the document that
  // was submitted, so if the draft has changed since, keep the newer edits and
  // take only the revision the server moved to.
  const latestDraft = useRef(draft);
  latestDraft.current = draft;
  const applyAnswer = (value: GlobalConfig, submittedYaml: string) => {
    if (latestDraft.current?.yaml === submittedYaml) apply(value);
    else setDraft((current) => current && { ...current, revision: value.revision });
  };
  const save = useMutation({
    mutationFn: async (body: ReturnType<typeof input>) => {
      // Keep submitted keys outside the mutation cache and independent of the page draft.
      const commitCredentials = credentials?.prepareCommit();
      const value = await api.saveGlobalConfig(body);
      // Keys are written only after their connections exist in the saved configuration.
      let keysFailed = false;
      if (commitCredentials) {
        try {
          await commitCredentials(Object.keys(object(object(value.effective.llm).providers)));
        } catch {
          keysFailed = true;
        }
      }
      return { value, keysFailed };
    },
    onSuccess: ({ value, keysFailed }, body) => {
      applyAnswer(value, body.yaml);
      setUncommitted(false);
      // Drop only the renames this save carried; one made while it ran still
      // belongs to the newer draft.
      const unsent = (current: Record<string, string>, sent: Record<string, string> = {}) =>
        Object.fromEntries(Object.entries(current).filter(([from, to]) => sent[from] !== to));
      setRenames((current) => unsent(current, body.model_renames));
      setProviderRenames((current) => unsent(current, body.provider_renames));
      if (credentials) void credentials.refresh();
      qc.setQueryData(["globalConfig"], value);
      for (const key of ["config", "models", "templates"])
        void qc.invalidateQueries({ queryKey: [key] });
      setCredentialsFailed(keysFailed);
      if (keysFailed) toast.error(t("settings.credentialsFailed"));
      else toast.success(t("settings.globalSaved"));
    },
  });
  const validate = useMutation({
    mutationFn: (body: ReturnType<typeof input>) => api.validateGlobalConfig(body),
    onSuccess: (value, body) => {
      setUncommitted((current) => current || draft?.yaml !== query.data?.yaml);
      applyAnswer(value, body.yaml);
      toast.success(t("settings.configurationIsValidButNotSavedYet"));
    },
  });
  const restore = useMutation({
    mutationFn: api.getGlobalDefaults,
    onSuccess: (value) => {
      setUncommitted(true);
      apply({ ...value, revision: draft!.revision });
      setRenames({});
      setProviderRenames({});
      setRegistryKey((value) => value + 1);
      credentials?.retain(Object.keys(object(object(value.effective.llm).providers)));
      toast.success(t("settings.defaultsLoaded"));
    },
  });
  const error = save.error || validate.error || restore.error;
  // Installing an app update is a state the page is in, so it disables the form
  // and the actions that would write configuration underneath the relaunch.
  // A request in flight is not: disabling for it applied the disabled face to
  // every control and stripped the hover state from the button being pressed.
  const pending = save.isPending || validate.isPending || restore.isPending;
  const disabled = !draft || yamlDirty || installing;
  const effective = draft?.effective || {};
  const llm = object(effective.llm);
  // Until the first load completes there is no draft to compare against, so
  // every derivation would read as "unsaved". Reporting that for one frame
  // both flashes the unsaved notice and, because the notice widens the action
  // bar, shifts the bar sideways before the load settles.
  const loaded = !!draft;
  const configSaved = loaded && draft.yaml === query.data?.yaml;
  const dirty = (loaded && !configSaved) || uncommitted || yamlDirty || credentialsPending;
  const change = (next: Document) => {
    if (draft)
      setDraft({
        ...draft,
        effective: next,
        yaml: JSON.stringify(next, null, 2),
      });
  };
  const changeLlm = (value: Document) => {
    change({ ...effective, llm: { ...value, preset: null } });
    credentials?.retain(Object.keys(object(value.providers)));
    setProviderRenames((current) =>
      Object.fromEntries(
        Object.entries(current).filter(([, id]) => id in object(value.providers)),
      ),
    );
    setRenames((current) =>
      Object.fromEntries(
        Object.entries(current).filter(([, id]) => id in object(value.models)),
      ),
    );
  };
  const rename = (group: RegistryGroup, oldId: string, newId: string) => {
    if (group === "providers" && credentials) {
      credentials.rename(oldId, newId);
      const next = { ...providerRenames };
      const original = Object.keys(next).find((id) => next[id] === oldId);
      if (original) {
        if (original === newId) delete next[original];
        else next[original] = newId;
      } else if (oldId in object(object(query.data?.effective.llm).providers)) {
        next[oldId] = newId;
      }
      setProviderRenames(next);
    }
    if (group === "models") {
      const next = { ...renames };
      const original = Object.keys(next).find((id) => next[id] === oldId);
      if (original) {
        if (original === newId) delete next[original];
        else next[original] = newId;
      } else if (
        oldId in object(object(query.data?.effective.llm).models) &&
        !(oldId in next)
      )
        next[oldId] = newId;
      setRenames(next);
    }
    changeLlm(renameRegistryId(llm, group, oldId, newId));
  };
  return (
    <>
      <ErrorNotice error={query.error || error} />
      {/* Distinct from the toast that announces the failure: this one stays, so
          it describes the state the keys are in rather than repeating the event. */}
      {credentialsFailed && (
        <ErrorNotice error={t("settings.credentialsFailedNotice")} />
      )}
      {/* Keep controls mounted so category navigation preserves local ID/key drafts. */}
      <div className={SETTINGS_CONTENT_CLASSNAME}>
        <div hidden={section !== "general"}>
          <LanguageSettings />
          <SettingsGroup title={t("settings.sectionExport")}>
            <ExportSettings
              config={effective}
              disabled={disabled}
              onField={(group, key, value) => change(workflowField(effective, group, key, value))}
            />
          </SettingsGroup>
        </div>
        <div hidden={section !== "providers"}>
          <SettingsSection id="provider-models" title={t("settings.sectionProviders")} grouped>
            <ProviderSettings
              key={registryKey}
              config={effective}
              kinds={caps?.providers || []}
              disabled={disabled}
              onChange={changeLlm}
              onRename={rename}
              onEditingChange={setEditingIds}
              saved={configSaved && !credentialsPending}
            />
          </SettingsSection>
        </div>
        <div hidden={section !== "defaults"}>
          <SettingsSection title={t("settings.sectionDefaultModels")} grouped>
            <SettingsGroup title={t("settings.modelTiers")}>
              <ModelSelection
                llm={llm}
                models={object(llm.models)}
                disabled={disabled}
                onChange={changeLlm}
              />
            </SettingsGroup>
          </SettingsSection>
        </div>
        <div hidden={section !== "workflow"}>
          <SettingsSection title={t("settings.sectionWorkflow")} grouped>
            <SettingsGroup title={t("settings.pipelineSteps")}>
              <WorkflowSettings
                scope="global"
                config={effective}
                disabled={disabled}
                onField={(group, key, value) => change(workflowField(effective, group, key, value))}
              />
            </SettingsGroup>
          </SettingsSection>
        </div>
        <div hidden={section !== "pdf"}>
          <PdfSettings disabled={installing} />
        </div>
        <div hidden={section !== "advanced"}>
          <SettingsSection id="advanced-yaml" title={t("settings.sectionAdvanced")} grouped>
            <SettingsGroup
              title={t("settings.advancedYamlConfiguration")}
              summary={yamlDirty ? t("settings.unsavedSummary") : undefined}
            >
              <div className={cn(cardSurface, "p-4")}>
                <Textarea
                  aria-label={t("settings.advancedYamlConfiguration")}
                  spellCheck={false}
                  className="min-h-editor font-mono text-xs"
                  value={draft?.yaml || ""}
                  disabled={!draft || pending || installing}
                  onChange={(event) => {
                    if (draft) {
                      setDraft({ ...draft, yaml: event.target.value });
                      setYamlDirty(true);
                    }
                  }}
                />
              </div>
            </SettingsGroup>
          </SettingsSection>
        </div>
        {Diagnostics && (
          <div hidden={section !== "diagnostics"}><Diagnostics /></div>
        )}
        {Updates && (
          <div hidden={section !== "updates"}>
            <LazyBoundary>
              <Updates
                blocked={!draft || !query.data || draft.yaml !== query.data.yaml ||
                  uncommitted || yamlDirty || editingIds || pending || mutations > 0 ||
                  Object.keys(renames).length > 0 || Object.keys(providerRenames).length > 0}
                onInstalling={setInstalling}
              />
            </LazyBoundary>
          </div>
        )}
      </div>
      <SettingsActionBar
        notices={
          yamlDirty || editingIds || dirty ? (
            <>
              {yamlDirty && (
                <Link to="/settings/advanced" state={location.state} className="block text-warning-text underline underline-offset-4">
                  {t("settings.advancedYamlHasUnvalidatedChangesValidateIt")}
                </Link>
              )}
              {editingIds && (
                <Link to="/settings/providers" state={location.state} className="block text-warning-text underline underline-offset-4">
                  {t("registry.finishRenaming")}
                </Link>
              )}
              {!yamlDirty && !editingIds && dirty && (
                <span className="text-muted-foreground">{t("settings.unsavedSummary")}</span>
              )}
            </>
          ) : undefined
        }
      >
        <Button
          variant="outline"
          disabled={!draft || installing}
          aria-busy={restore.isPending || undefined}
          onClick={() => { if (!pending) restore.mutate(); }}
        >
          {t("settings.restoreDefaults")}
        </Button>
        {section === "advanced" && (
          <Button
            variant="outline"
            disabled={!draft || editingIds || installing}
            aria-busy={validate.isPending || undefined}
            onClick={() => { if (!pending) validate.mutate(input()); }}
          >
            {t("settings.validateConfiguration")}
          </Button>
        )}
        <Button
          disabled={!draft || editingIds || installing}
          aria-busy={save.isPending || undefined}
          onClick={() => { if (!pending) save.mutate(input()); }}
        >
          {t("settings.saveConfiguration")}
        </Button>
      </SettingsActionBar>
    </>
  );
}
