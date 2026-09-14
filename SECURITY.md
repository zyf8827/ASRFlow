# 安全说明 (Security Policy)

## 默认无鉴权提醒

> [!WARNING]
> ASRFlow 的原生 WebSocket 与 HTTP 服务**默认不包含用户身份鉴权**。
> 如果直接将服务端口暴露在公共网络中，任何客户端均可连接推流并占用推理算力。
> 
> 建议将服务运行在受信局域网内，或在前端反向代理（如 Nginx、Envoy、API Gateway）配置身份认证（如 Token / JWT）与请求限流。

## 报告安全问题

如果您在本项目中发现了安全问题或漏洞，请通过以下方式反馈：
- 在 GitHub 仓库中提交 [Issue](https://github.com/asrflow/asrflow/issues)；
- 或直接联系项目维护者。

我们会尽快跟进处理。
