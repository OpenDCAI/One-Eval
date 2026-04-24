import { useMemo, useState, useEffect } from "react";
import axios from "axios";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { 
  Plus, Save, Database, Cloud, KeyRound, Trash2, PlugZap, 
  ChevronDown, CheckCircle2
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useLang } from "@/lib/i18n";

interface ModelConfig {
  name: string;
  path: string;
  is_api?: boolean;
  api_url?: string;
  api_key?: string;
}

interface SettingsCardProps {
  title: string;
  description: string;
  icon: React.ElementType;
  iconColorClass?: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const SettingsCard = ({ 
  title, 
  description, 
  icon: Icon, 
  iconColorClass = "bg-primary/10 text-primary", 
  children, 
  defaultOpen = false 
}: SettingsCardProps) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <Card className="overflow-hidden border-slate-200 shadow-sm hover:shadow-md transition-all duration-300">
      <CardHeader 
        className="cursor-pointer bg-slate-50/30 hover:bg-slate-50/80 transition-colors p-6"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`p-2.5 rounded-xl ${iconColorClass}`}>
              <Icon className="w-6 h-6" />
            </div>
            <div>
              <CardTitle className="text-lg font-semibold text-slate-900">{title}</CardTitle>
              <CardDescription className="text-slate-500 mt-1">{description}</CardDescription>
            </div>
          </div>
          <ChevronDown 
            className={`w-5 h-5 text-slate-400 transition-transform duration-300 ${isOpen ? "rotate-180" : ""}`} 
          />
        </div>
      </CardHeader>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
          >
            <div className="border-t border-slate-100">
              <CardContent className="p-6 pt-6 space-y-6">
                {children}
              </CardContent>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
};

export const Settings = () => {
  const { lang } = useLang();
  const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [newModel, setNewModel] = useState<ModelConfig>({ name: "", path: "", is_api: false, api_url: "", api_key: "" });
  const [loading, setLoading] = useState(false);
  const [apiBaseUrl] = useState(() => localStorage.getItem("oneEval.apiBaseUrl") || "http://localhost:8111");
  const [hfEndpoint, setHfEndpoint] = useState("https://hf-mirror.com");
  const [hfToken, setHfToken] = useState("");
  const [hfTokenSet, setHfTokenSet] = useState(false);
  const [savingHf, setSavingHf] = useState(false);
  const [agentBaseUrl, setAgentBaseUrl] = useState("http://123.129.219.111:3000/v1");
  const [agentModel, setAgentModel] = useState("gpt-4o");
  const [agentApiKeyInput, setAgentApiKeyInput] = useState("");
  const [agentApiKeySet, setAgentApiKeySet] = useState(false);
  const [agentTimeoutS, setAgentTimeoutS] = useState(15);
  const [savingAgent, setSavingAgent] = useState(false);
  const [testingAgent, setTestingAgent] = useState(false);
  const [agentTestResult, setAgentTestResult] = useState<string | null>(null);
  const [showAgentSuccess, setShowAgentSuccess] = useState(false);
  const [testingModelPath, setTestingModelPath] = useState<string | null>(null);
  const [modelTestMsg, setModelTestMsg] = useState<Record<string, string>>({});

  const agentUrlPresets = useMemo(
    () => [
      { label: "yuchaAPI", value: "http://123.129.219.111:3000/v1/chat/completions" },
      { label: "OpenAI", value: "https://api.openai.com/v1" },
      { label: "OpenRouter", value: "https://openrouter.ai/api/v1" },
      { label: "Apiyi (OpenAI Compatible)", value: "https://api.apiyi.com/v1" },
      { label: "Custom...", value: "__custom__" },
    ],
    []
  );
  const agentUrlPresetValue = useMemo(() => {
    const hit = agentUrlPresets.find((p) => p.value === agentBaseUrl);
    return hit ? hit.value : "__custom__";
  }, [agentUrlPresets, agentBaseUrl]);

  const isValidHttpUrl = (u: string) => {
    try {
      const parsed = new URL(u);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
      return false;
    }
  };

  useEffect(() => {
    if (!isValidHttpUrl(apiBaseUrl)) return;
    fetchModels();
    fetchHfConfig();
    fetchAgentConfig();
  }, [apiBaseUrl]);

  const fetchModels = async () => {
    try {
      const res = await axios.get(`${apiBaseUrl}/api/models`);
      setModels(res.data);
    } catch (e) {
      console.error("Failed to fetch models", e);
    }
  };

  const handleSaveModel = async () => {
    if (!newModel.name || !newModel.path) return;
    setLoading(true);
    try {
      const payload: any = { name: newModel.name, path: newModel.path };
      if (newModel.is_api) {
        payload.is_api = true;
        payload.api_url = newModel.api_url;
        payload.api_key = newModel.api_key;
      }
      await axios.post(`${apiBaseUrl}/api/models`, payload);
      setModels([...models, payload]);
      setNewModel({ name: "", path: "", is_api: false, api_url: "", api_key: "" });
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const fetchHfConfig = async () => {
    try {
      const res = await axios.get(`${apiBaseUrl}/api/config/hf`);
      setHfEndpoint(res.data.endpoint || "https://hf-mirror.com");
      setHfTokenSet(Boolean(res.data.token_set));
    } catch (e) {
      setHfEndpoint("https://hf-mirror.com");
      setHfTokenSet(false);
    }
  };

  const handleSaveHfConfig = async () => {
    setSavingHf(true);
    try {
      const payload: any = { endpoint: hfEndpoint };
      if (hfToken.trim()) payload.token = hfToken;
      const res = await axios.post(`${apiBaseUrl}/api/config/hf`, payload);
      setHfEndpoint(res.data.endpoint || hfEndpoint);
      setHfTokenSet(Boolean(res.data.token_set));
      setHfToken("");
    } catch (e) {
      console.error(e);
    }
    setSavingHf(false);
  };

  const handleClearHfToken = async () => {
    setSavingHf(true);
    try {
      const res = await axios.post(`${apiBaseUrl}/api/config/hf`, { clear_token: true });
      setHfEndpoint(res.data.endpoint || hfEndpoint);
      setHfTokenSet(Boolean(res.data.token_set));
      setHfToken("");
    } catch (e) {
      console.error(e);
    }
    setSavingHf(false);
  };

  const fetchAgentConfig = async () => {
    try {
      const res = await axios.get(`${apiBaseUrl}/api/config/agent`);
      setAgentBaseUrl(res.data.base_url || "http://123.129.219.111:3000/v1");
      setAgentModel(res.data.model || "gpt-4o");
      setAgentApiKeySet(Boolean(res.data.api_key_set));
      setAgentTimeoutS(Number(res.data.timeout_s || 15));
      setAgentApiKeyInput("");
    } catch (e) {
      setAgentBaseUrl("http://123.129.219.111:3000/v1");
      setAgentModel("gpt-4o");
      setAgentApiKeySet(false);
      setAgentTimeoutS(15);
      setAgentApiKeyInput("");
    }
  };

  const handleSaveAgentConfig = async () => {
    setSavingAgent(true);
    try {
      const payload: any = {
        base_url: agentBaseUrl,
        model: agentModel,
        timeout_s: agentTimeoutS,
      };
      if (agentApiKeyInput.trim()) payload.api_key = agentApiKeyInput.trim();
      const res = await axios.post(`${apiBaseUrl}/api/config/agent`, payload);
      setAgentBaseUrl(res.data.base_url || agentBaseUrl);
      setAgentModel(res.data.model || agentModel);
      setAgentApiKeySet(Boolean(res.data.api_key_set));
      setAgentTimeoutS(Number(res.data.timeout_s || agentTimeoutS));
      // Keep the input and result visible so user knows what happened
      // setAgentApiKeyInput(""); 
      // setAgentTestResult(null);
      setShowAgentSuccess(true);
      setTimeout(() => setShowAgentSuccess(false), 3000);
    } catch (e) {
      console.error(e);
    }
    setSavingAgent(false);
  };

  const handleClearAgentApiKey = async () => {
    setSavingAgent(true);
    try {
      const res = await axios.post(`${apiBaseUrl}/api/config/agent`, { clear_api_key: true });
      setAgentApiKeySet(Boolean(res.data.api_key_set));
      setAgentApiKeyInput("");
      setAgentTestResult(null);
    } catch (e) {
      console.error(e);
    }
    setSavingAgent(false);
  };

  const handleTestAgentConnection = async () => {
    setTestingAgent(true);
    setAgentTestResult(null);
    try {
      const payload: any = {
        base_url: agentBaseUrl,
        model: agentModel,
        timeout_s: agentTimeoutS,
      };
      // Send the currently input API key if it's not empty, otherwise don't send it (let backend use saved key)
      // Actually, if we want to test "what I just typed", we should send it even if empty string?
      // But if user has a saved key and clears the input, maybe they mean "use saved"?
      // No, consistent UX: "Test" tests the *current form values*.
      // If user clears the input, they might mean "no auth".
      // However, for security, we don't auto-fill the input with the saved key.
      // So if input is empty, and there IS a saved key (agentApiKeySet is true), we probably want to use the saved key.
      // If input is NOT empty, use the input.
      if (agentApiKeyInput.trim()) {
        payload.api_key = agentApiKeyInput.trim();
      } else if (!agentApiKeySet) {
          // No saved key, and no input key -> send empty to override any potential default? 
          // Backend falls back to saved config if req.api_key is None.
          // If we send "", backend treats it as empty key.
          // If we don't send it, backend uses saved key.
          // If there is NO saved key (agentApiKeySet false), backend has None.
          // So if input is empty and not saved, we can just send nothing.
      } else {
         // Input empty, but key is saved. 
         // We should NOT send api_key field so backend uses the saved one.
      }

      const res = await axios.post(`${apiBaseUrl}/api/config/agent/test`, payload);
      if (res.data.ok) {
        setAgentTestResult(`OK (${res.data.mode})`);
      } else {
        const code = res.data.status_code ? ` [${res.data.status_code}]` : "";
        setAgentTestResult(`${tt("失败", "FAILED")}${code}: ${res.data.detail}`);
      }
    } catch (e) {
      setAgentTestResult(`${tt("失败", "FAILED")}: ${tt("请求异常", "request error")}`);
    }
    setTestingAgent(false);
  };

  const handleTestModelLoad = async (modelPath: string) => {
    const path = (modelPath || "").trim();
    if (!path) return;
    setTestingModelPath(path);
    setModelTestMsg((prev) => ({ ...prev, [path]: tt("测试中...", "Testing...") }));
    try {
      const res = await axios.post(`${apiBaseUrl}/api/models/test_load`, { model_path: path });
      const ok = !!res.data?.ok;
      setModelTestMsg((prev) => ({
        ...prev,
        [path]: ok ? tt("加载成功", "Load passed") : tt("加载失败", "Load failed"),
      }));
    } catch (e: any) {
      const detail = e?.response?.data?.detail || tt("请求异常", "request error");
      setModelTestMsg((prev) => ({ ...prev, [path]: `${tt("加载失败", "Load failed")}: ${detail}` }));
    }
    setTestingModelPath(null);
  };

  return (
    <div className="p-12 max-w-[1600px] mx-auto space-y-8">
      <div className="space-y-2 mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">{tt("设置", "Settings")}</h1>
        <p className="text-slate-500 text-lg">{tt("配置评测环境与模型注册表。", "Configure your evaluation environment and model registry.")}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 items-start">
        {/* 1. One-Eval Backend (Hidden) */}

        {/* 2. Agent Server */}
        <SettingsCard
          title={tt("Agent 服务", "Agent Server")}
          description={tt("配置用于评测流程的 LLM 提供方（如 OpenAI、vLLM 等）。", "Configure the LLM provider (e.g. OpenAI, vLLM, etc.) used for evaluation.")}
          icon={PlugZap}
          iconColorClass="bg-violet-500/10 text-violet-600"
          defaultOpen={true}
        >
          <div className="space-y-6">
            <div className="space-y-2">
              <Label>{tt("服务地址", "Provider URL")}</Label>
              <div className="grid grid-cols-1 gap-2">
                <select
                  value={agentUrlPresetValue}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (v !== "__custom__") setAgentBaseUrl(v);
                  }}
                  className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                >
                  {agentUrlPresets.map((p) => (
                    <option key={p.value} value={p.value}>
                      {p.label}
                    </option>
                  ))}
                </select>
                <Input
                  value={agentBaseUrl}
                  onChange={(e) => setAgentBaseUrl(e.target.value)}
                  placeholder={tt("https://.../v1 或 https://.../v1/chat/completions", "https://.../v1  or  https://.../v1/chat/completions")}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>{tt("模型", "Model")}</Label>
                <select
                  value={agentModel}
                  onChange={(e) => setAgentModel(e.target.value)}
                  className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                >
                  <option value="gpt-4o">gpt-4o</option>
                  <option value="gpt-5.1">gpt-5.1</option>
                  <option value="gpt-5.2">gpt-5.2</option>
                  <option value="deepseek-v3">deepseek-v3</option>
                  <option value="deepseek-r1">deepseek-r1</option>
                </select>
              </div>
              <div className="space-y-2">
                <Label>{tt("超时（秒）", "Timeout (s)")}</Label>
                <Input
                  type="number"
                  value={agentTimeoutS}
                  onChange={(e) => setAgentTimeoutS(Number(e.target.value || 15))}
                  className="border-slate-200"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>API Key</Label>
                {agentApiKeySet && <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">{tt("密钥已保存", "Key Saved")}</span>}
              </div>
              <Input
                type="password"
                value={agentApiKeyInput}
                onChange={(e) => setAgentApiKeyInput(e.target.value)}
                placeholder={tt("sk-...（出于安全考虑不会自动回填）", "sk-... (won't be auto-filled for security)")}
              />
            </div>

            <div className="flex gap-3">
              <Button variant="outline" className="flex-1" onClick={handleTestAgentConnection} disabled={testingAgent}>
                {testingAgent ? tt("测试中...", "Testing...") : tt("测试连接", "Test Connection")}
              </Button>
              <Button
                className={`flex-1 text-white transition-all duration-300 ${
                  showAgentSuccess 
                    ? "bg-emerald-600 hover:bg-emerald-700 shadow-emerald-600/20" 
                    : "bg-slate-900 hover:bg-slate-800"
                }`}
                onClick={handleSaveAgentConfig}
                disabled={savingAgent}
              >
                {savingAgent ? (
                  tt("保存中...", "Saving...")
                ) : showAgentSuccess ? (
                  <><CheckCircle2 className="w-4 h-4 mr-2" /> {tt("已保存", "Saved!")}</>
                ) : (
                  tt("保存配置", "Save Configuration")
                )}
              </Button>
            </div>

            <div className="flex items-center justify-between pt-2">
               <Button variant="ghost" size="sm" className="text-red-500 hover:text-red-600 hover:bg-red-50" onClick={handleClearAgentApiKey} disabled={savingAgent}>
                <Trash2 className="w-4 h-4 mr-2" />
                {tt("清除 API Key", "Clear API Key")}
              </Button>
              {agentTestResult && (
                <div className={`text-xs px-3 py-1.5 rounded-md font-mono ${agentTestResult.startsWith("OK") ? "bg-emerald-50 text-emerald-700 border border-emerald-200" : "bg-red-50 text-red-700 border border-red-200"}`}>
                  {agentTestResult}
                </div>
              )}
            </div>
          </div>
        </SettingsCard>

        {/* 3. HuggingFace */}
        <SettingsCard
          title={tt("HuggingFace 配置", "HuggingFace Configuration")}
          description={tt("配置 HF 镜像地址和访问令牌，用于下载模型与数据集。", "Configure HF mirror endpoint and access token for downloading models/datasets.")}
          icon={Cloud}
          iconColorClass="bg-amber-500/10 text-amber-600"
          defaultOpen={false}
        >
          <div className="space-y-6">
            <div className="space-y-2">
              <Label>{tt("HF 镜像地址", "HF Mirror Endpoint")}</Label>
              <Input
                value={hfEndpoint}
                onChange={(e) => setHfEndpoint(e.target.value)}
                placeholder="https://hf-mirror.com"
              />
              <p className="text-xs text-slate-500">{tt("默认值：https://hf-mirror.com", "Default: https://hf-mirror.com")}</p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>HF Token</Label>
                {hfTokenSet && <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">{tt("令牌已保存", "Token Saved")}</span>}
              </div>
              <Input
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder="hf_..."
              />
              <p className="text-xs text-slate-500">
                {tt("留空则保留当前已保存令牌。", "Leave empty to keep the currently saved token.")}
              </p>
            </div>

            <div className="flex gap-3">
              <Button
                className="flex-1 text-white bg-slate-900 hover:bg-slate-800"
                onClick={handleSaveHfConfig}
                disabled={savingHf}
              >
                {savingHf ? tt("保存中...", "Saving...") : <><KeyRound className="w-4 h-4 mr-2" /> {tt("保存 HF 配置", "Save HF Config")}</>}
              </Button>
              <Button
                variant="outline"
                className="flex-1 text-red-500 hover:text-red-600 hover:bg-red-50"
                onClick={handleClearHfToken}
                disabled={savingHf}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                {tt("清除令牌", "Clear Token")}
              </Button>
            </div>
          </div>
        </SettingsCard>

        {/* 4. Model Registry */}
        <SettingsCard
          title={tt("目标模型注册表", "Target Model Registry")}
          description={tt("注册你希望参与评测的本地或远端模型。", "Register local or remote models that you want to evaluate.")}
          icon={Database}
          iconColorClass="bg-pink-500/10 text-pink-600"
          defaultOpen={true}
        >
          <div className="space-y-6">
            {/* Add New */}
            <div className="p-5 border border-slate-200 rounded-xl bg-slate-50/50 space-y-4">
              <h4 className="text-sm font-semibold flex items-center gap-2 text-slate-800">
                <Plus className="w-4 h-4" /> {tt("新增模型", "Add New Model")}
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>{tt("模型名称", "Model Name")}</Label>
                  <Input 
                    placeholder={tt("例如：Qwen2.5-7B-Instruct", "e.g. Qwen2.5-7B-Instruct")} 
                    value={newModel.name}
                    onChange={e => setNewModel({...newModel, name: e.target.value})}
                    className="bg-white"
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                      <Label>{tt("模型路径 / API 模型名", "Model Path / API Model Name")}</Label>
                      <div className="flex items-center gap-1.5">
                          <input 
                             type="checkbox" 
                             id="apiModelSetting"
                             checked={newModel.is_api} 
                             onChange={(e) => setNewModel({...newModel, is_api: e.target.checked})} 
                          />
                          <label htmlFor="apiModelSetting" className="text-[10px] text-blue-600 cursor-pointer font-medium">{tt("API 模型", "API Model")}</label>
                      </div>
                  </div>
                  <Input 
                    placeholder={newModel.is_api ? "gpt-4o" : "/mnt/models/..."} 
                    value={newModel.path}
                    onChange={e => setNewModel({...newModel, path: e.target.value})}
                    className="bg-white"
                  />
                </div>
                
                {newModel.is_api && (
                    <>
                        <div className="space-y-2">
                            <Label>API URL</Label>
                            <Input 
                                placeholder="https://api.openai.com/v1/chat/completions" 
                                value={newModel.api_url}
                                onChange={e => setNewModel({...newModel, api_url: e.target.value})}
                                className="bg-white"
                            />
                        </div>
                        <div className="space-y-2">
                            <Label>API Key</Label>
                            <Input 
                                type="password"
                                placeholder="sk-..." 
                                value={newModel.api_key}
                                onChange={e => setNewModel({...newModel, api_key: e.target.value})}
                                className="bg-white"
                            />
                        </div>
                    </>
                )}
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Button
                  variant="outline"
                  onClick={() => handleTestModelLoad(newModel.path)}
                  disabled={!newModel.path.trim() || !!testingModelPath || newModel.is_api}
                  className="w-full"
                >
                  {testingModelPath === newModel.path.trim() ? tt("测试中...", "Testing...") : tt("测试加载本地模型", "Test Local Model Load")}
                </Button>
                <Button onClick={handleSaveModel} disabled={loading} className="w-full text-white bg-slate-900 hover:bg-slate-800">
                  {loading ? tt("保存中...", "Saving...") : <><Save className="w-4 h-4 mr-2"/> {tt("加入注册表", "Add to Registry")}</>}
                </Button>
              </div>
              {newModel.path.trim() && modelTestMsg[newModel.path.trim()] && (
                <div className={`text-xs px-3 py-2 rounded border ${modelTestMsg[newModel.path.trim()].includes(tt("加载成功", "Load passed")) ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-red-50 text-red-700 border-red-200"}`}>
                  {modelTestMsg[newModel.path.trim()]}
                </div>
              )}
            </div>

            {/* List */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-slate-500 uppercase tracking-wider text-xs">{tt("已注册模型", "Registered Models")}</h4>
              {models.length === 0 && (
                <div className="text-center py-8 border-2 border-dashed border-slate-200 rounded-xl">
                  <p className="text-sm text-slate-400">{tt("暂未注册模型。", "No models registered yet.")}</p>
                </div>
              )}
              <div className="grid grid-cols-1 gap-3">
                {models.map((m, i) => (
                  <div key={i}>
                    <div className="flex items-center justify-between p-4 rounded-xl border border-slate-100 bg-white hover:border-slate-200 hover:shadow-sm transition-all">
                      <div className="flex-1 min-w-0 mr-4">
                        <div className="font-semibold text-slate-900 flex items-center gap-2">
                          {m.name}
                          {m.is_api && <span className="text-[10px] bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded font-bold uppercase">API</span>}
                        </div>
                        <div className="text-xs text-slate-500 truncate font-mono mt-1" title={m.path}>{m.path}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        {!m.is_api && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleTestModelLoad(m.path)}
                              disabled={testingModelPath === m.path}
                            >
                              {testingModelPath === m.path ? tt("测试中...", "Testing...") : tt("测试加载", "Test Load")}
                            </Button>
                        )}
                      </div>
                    </div>
                    {modelTestMsg[m.path] && (
                      <div className={`-mt-2 mb-1 text-xs px-3 py-2 rounded border ${modelTestMsg[m.path].includes(tt("加载成功", "Load passed")) ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-red-50 text-red-700 border-red-200"}`}>
                        {modelTestMsg[m.path]}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </SettingsCard>

      </div>
    </div>
  );
};

export default Settings;
