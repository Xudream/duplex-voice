// duplex-voice App 客户端（P1：webview 嵌入模式）
//
// P1 架构：Flutter 壳 + flutter_inappwebview 加载服务端页面
//   - 服务端地址首次配置（Server URL 输入页，存 SharedPreferences）
//   - 页面内的 VAD/打断/时延面板逻辑零改动复用
//   - 麦克风权限：Android RecordAudio / iOS NSMicrophoneUsageDescription
// P2：原生音频层（record + just_audio + WS /api/stream）替换 webview 采集，
//     浏览器 AEC -> 系统 AEC（AcousticEchoCanceler / AVAudioSession voiceChat）
import 'package:flutter/material.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() => runApp(const DuplexVoiceApp());

class DuplexVoiceApp extends StatelessWidget {
  const DuplexVoiceApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'duplex-voice',
        theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
        home: const HomePage(),
      );
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String? serverUrl;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    var url = prefs.getString('server_url');
    // 默认地址：正式域名（有真实证书）；用户可改
    if (url == null || url.isEmpty) {
      url = 'https://nofishly.xyz';
      await prefs.setString('server_url', url);
    }
    if (!mounted) return;
    setState(() => serverUrl = url);
  }

  @override
  Widget build(BuildContext context) {
    if (serverUrl == null || serverUrl!.isEmpty) {
      return const ServerSetupPage();
    }
    return VoicePage(url: serverUrl!);
  }
}

/// 首次配置：输入服务端地址（如 https://192.168.1.100:8787）
class ServerSetupPage extends StatefulWidget {
  const ServerSetupPage({super.key});
  @override
  State<ServerSetupPage> createState() => _ServerSetupPageState();
}

class _ServerSetupPageState extends State<ServerSetupPage> {
  final _controller = TextEditingController();

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('连接服务端')),
        body: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              TextField(
                controller: _controller,
                decoration: const InputDecoration(
                    labelText: '服务端地址',
                    hintText: 'https://192.168.1.100:8787'),
              ),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: () async {
                  final prefs = await SharedPreferences.getInstance();
                  await prefs.setString('server_url', _controller.text.trim());
                  if (context.mounted) {
                    Navigator.of(context).pushReplacement(MaterialPageRoute(
                        builder: (_) => VoicePage(url: _controller.text.trim())));
                  }
                },
                child: const Text('连接'),
              ),
            ],
          ),
        ),
      );
}

/// 语音主页面：webview 嵌入服务端 index.html（P1）
class VoicePage extends StatefulWidget {
  const VoicePage({super.key, required this.url});
  final String url;
  @override
  State<VoicePage> createState() => _VoicePageState();
}

class _VoicePageState extends State<VoicePage> {
  @override
  Widget build(BuildContext context) => Scaffold(
        body: InAppWebView(
          initialUrlRequest: URLRequest(url: WebUri(widget.url)),
          initialSettings: InAppWebViewSettings(
            mediaPlaybackRequiresUserGesture: false,   // 免点按自动播放 TTS
            allowsInlineMediaPlayback: true,
          ),
          // 服务端为 IP 自签证书（Caddy tls internal）：放行，不弹系统证书错误页
          // onReceivedServerTrustAuthRequest: (controller, challenge) async =>
          //     ServerTrustAuthResponse(
          //         action: ServerTrustAuthResponseAction.PROCEED),
          onPermissionRequest: (controller, request) async =>
              PermissionResponse(
                  resources: request.resources,
                  action: PermissionResponseAction.GRANT),
        ),
      );
}
