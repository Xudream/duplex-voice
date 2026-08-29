// duplex-voice App 原生语音页（P2）
//
// 架构：原生音频层直连服务端 WS /api/stream（替代 P1 webview 采集）
//   - 采集：record 插件（16kHz mono PCM16，Android AcousticEchoCanceler / iOS voiceChat）
//   - 播放：just_audio（服务端 TTS wav 下发，AudioSession 播放入口）
//   - 通信：web_socket_channel（wss://host/api/stream，hello 带登录会话 token）
//   - 判停：轻量能量 VAD（RMS 门限 + 静音超时）→ 发 is_final；按住说话 = 松开判停
//   - 服务端语义 VAD（文本/音频）不变，仍走快慢融合 + TTS
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Uint8List;
import 'package:http/http.dart' as http;
import 'package:just_audio/just_audio.dart';
import 'package:record/record.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

/// 原生语音页（P2）
class NativeVoicePage extends StatefulWidget {
  const NativeVoicePage({super.key, required this.serverUrl});
  final String serverUrl;

  @override
  State<NativeVoicePage> createState() => _NativeVoicePageState();
}

class _NativeVoicePageState extends State<NativeVoicePage> {
  // ---- 连接 ----
  WebSocketChannel? _ws;
  StreamSubscription? _wsSub;
  bool _connected = false;
  String _connState = '未连接';

  // ---- 采集 ----
  final _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _recSub;
  final List<int> _pcmBuf = [];        // 16k s16le 累积（切段发送）
  bool _listening = false;             // 正在采集（按住说话/持续对话激活）
  bool _sending = false;               // 正在发本轮音频（等待回复）

  // ---- 能量 VAD（持续对话判停）----
  static const int _chunkBytes = 10240;   // 320ms @16k*2B
  static const double _vadThreshold = 0.012;  // RMS 门限（与 web 端段级能量下限一致）
  int _silenceMs = 0;                     // 当前静音累积
  static const int _silenceTimeoutMs = 1200;  // 静音 1.2s → 判停

  // ---- 播放 ----
  final _player = AudioPlayer();

  // ---- UI 状态 ----
  final List<_Msg> _msgs = [];
  String _statusLine = '就绪';
  final _scrollCtrl = ScrollController();
  bool _contMode = false;   // 持续对话（自动判停）vs 按住说话

  // ---- 登录状态（原生页独立登录，token 存 SharedPreferences）----
  bool _loggedIn = false;
  bool _loginBusy = false;
  final _userCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  String _loginErr = '';

  @override
  void initState() {
    super.initState();
    _checkLogin();
  }

  Future<void> _checkLogin() async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('auth_token') ?? '';
    setState(() => _loggedIn = token.isNotEmpty);
    if (token.isNotEmpty) _connect();
  }

  Future<void> _doLogin() async {
    final u = _userCtrl.text.trim();
    final p = _passCtrl.text;
    if (u.isEmpty || p.isEmpty) {
      setState(() => _loginErr = '请输入用户名和密码');
      return;
    }
    setState(() { _loginBusy = true; _loginErr = ''; });
    try {
      final resp = await http.post(
        Uri.parse('${widget.serverUrl}/api/login'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'username': u, 'password': p}),
      ).timeout(const Duration(seconds: 20));
      final d = jsonDecode(utf8.decode(resp.bodyBytes)) as Map<String, dynamic>;
      if (resp.statusCode == 200 && d['token'] != null) {
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('auth_token', d['token'] as String);
        setState(() => _loggedIn = true);
        _connect();
      } else {
        setState(() => _loginErr = (d['msg'] ?? '登录失败') as String);
      }
    } on TimeoutException {
      setState(() => _loginErr = '连接超时');
    } catch (e) {
      setState(() => _loginErr = '网络错误: $e');
    } finally {
      if (mounted) setState(() => _loginBusy = false);
    }
  }

  @override
  void dispose() {
    _wsSub?.cancel();
    _recSub?.cancel();
    _ws?.sink.close();
    _player.dispose();
    _userCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
  }

  // ==================== WS 连接 ====================
  Future<void> _connect() async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('auth_token') ?? '';
    if (token.isEmpty) {
      setState(() => _connState = '未登录');
      return;
    }
    final uri = widget.serverUrl
        .replaceFirst('https://', 'wss://')
        .replaceFirst('http://', 'ws://');
    try {
      final ws = WebSocketChannel.connect(Uri.parse('$uri/api/stream'));
      _ws = ws;
      _wsSub = ws.stream.listen(_onWsMsg, onError: (e) {
        _setConn(false, '连接错误: $e');
      }, onDone: () => _setConn(false, '连接断开'));
      ws.sink.add(jsonEncode({
        'type': 'hello',
        'token': token,
        'client_id': 'native_${DateTime.now().millisecondsSinceEpoch % 100000}',
      }));
      // 连接超时兜底（10s 未 ready → 提示）
      Timer(const Duration(seconds: 10), () {
        if (!_connected && mounted) _setConn(false, '连接超时');
      });
    } catch (e) {
      _setConn(false, '连接失败: $e');
    }
  }

  void _setConn(bool ok, String status) {
    if (!mounted) return;
    setState(() {
      _connected = ok;
      _connState = status;
      _statusLine = status;
    });
  }

  void _onWsMsg(dynamic raw) {
    if (raw is! String) return;
    final m = jsonDecode(raw) as Map<String, dynamic>;
    switch (m['type']) {
      case 'ready':
        _setConn(true, '已连接 (${m['vad_mode'] ?? '-'} / ${m['fusion'] ?? '-'})');
        break;
      case 'auth_error':
        _setConn(false, '鉴权失败，请重新登录');
        break;
      case 'asr_partial':
        _addMsg(_Msg(kind: 'asr', text: m['text'] ?? '', partial: true));
        break;
      case 'asr_final':
        _addMsg(_Msg(kind: 'asr', text: m['text'] ?? ''));
        break;
      case 'vad':
        _statusLine = 'VAD: ${m['state']}';
        setState(() {});
        break;
      case 'fast':
        _addMsg(_Msg(kind: 'fast', text: m['text'] ?? '', ms: m['ms']));
        break;
      case 'slow_first':
        _addMsg(_Msg(kind: 'slow', text: m['text'] ?? '', ms: m['ms']));
        break;
      case 'slow_delta':
        if (_msgs.isNotEmpty && _msgs.last.kind == 'slow') {
          _msgs.last.text += m['text'] ?? '';
          setState(() {});
        } else {
          _addMsg(_Msg(kind: 'slow', text: m['text'] ?? ''));
        }
        break;
      case 'tts':
        _playTts(m['b64'] ?? '');
        break;
      case 'latency':
        final parts = <String>[];
        if (m['fast_ms'] != null) parts.add('快 ${m['fast_ms']}ms');
        if (m['slow_first_ms'] != null) parts.add('慢首 ${m['slow_first_ms']}ms');
        if (m['asr_ms'] != null) parts.add('ASR ${m['asr_ms']}ms');
        if (m['tts_ms'] != null) parts.add('TTS ${m['tts_ms']}ms');
        _statusLine = '⏱ ${parts.join(' · ')}';
        setState(() {});
        break;
      case 'done':
        _sending = false;
        _statusLine = '本轮完成';
        setState(() {});
        break;
      case 'error':
        _sending = false;
        _addMsg(_Msg(kind: 'err', text: m['message'] ?? 'error'));
        break;
    }
  }

  // ==================== TTS 播放 ====================
  Future<void> _playTts(String b64) async {
    try {
      final bytes = base64Decode(b64);
      final dir = Directory.systemTemp;
      final file = File('${dir.path}/tts_${DateTime.now().millisecondsSinceEpoch}.wav');
      await file.writeAsBytes(bytes);
      await _player.setFilePath(file.path);
      _player.play();
    } catch (e) {
      _addMsg(_Msg(kind: 'err', text: 'TTS 播放失败: $e'));
    }
  }

  // ==================== 采集 ====================
  Future<void> _startCapture() async {
    if (_listening) return;
    try {
      final hasPerm = await _recorder.hasPermission();
      if (!hasPerm) {
        _addMsg(_Msg(kind: 'err', text: '无麦克风权限'));
        return;
      }
      _pcmBuf.clear();
      _silenceMs = 0;
      // Android: echoCancel → AcousticEchoCanceler（系统 AEC）；iOS: voiceChat 会话
      final stream = await _recorder.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        echoCancel: true,
        noiseSuppress: true,
        autoGain: false,
      ));
      _recSub = stream.listen((chunk) {
        _pcmBuf.addAll(chunk);
        if (_contMode) _energyVad(chunk);
      });
      setState(() {
        _listening = true;
        _statusLine = _contMode ? '持续对话中…' : '聆听中…';
      });
    } catch (e) {
      _addMsg(_Msg(kind: 'err', text: '采集启动失败: $e'));
    }
  }

  Future<void> _stopCapture({bool finalize = true}) async {
    if (!_listening) return;
    _recSub?.cancel();
    _recSub = null;
    try {
      await _recorder.stop();
    } catch (_) {}
    setState(() => _listening = false);
    if (finalize) _flushAudio(isFinal: true);
  }

  /// 能量 VAD：块级 RMS 超门限视为语音，静音持续超时 → 判停发 is_final
  void _energyVad(Uint8List chunk) {
    double rms = 0;
    for (int i = 0; i + 1 < chunk.length; i += 2) {
      final s = (chunk[i] | (chunk[i + 1] << 8)).toSigned(16) / 32768.0;
      rms += s * s;
    }
    rms = sqrt(rms / max(1, chunk.length ~/ 2));
    if (rms > _vadThreshold) {
      _silenceMs = 0;
    } else {
      _silenceMs += (chunk.length ~/ _chunkBytes) * 320;
      if (_silenceMs >= _silenceTimeoutMs && _pcmBuf.isNotEmpty) {
        _silenceMs = 0;
        _flushAudio(isFinal: true);
      }
    }
  }

  /// 发送累积音频：分块（320ms/块）上行，末块 is_final=true
  void _flushAudio({required bool isFinal}) {
    if (_pcmBuf.isEmpty || _ws == null || !_connected) return;
    final all = Uint8List.fromList(_pcmBuf);
    _pcmBuf.clear();
    _sending = true;
    for (int off = 0; off < all.length; off += _chunkBytes) {
      final end = min(off + _chunkBytes, all.length);
      final chunk = all.sublist(off, end);
      final isLast = isFinal && end == all.length;
      _ws!.sink.add(jsonEncode({
        'type': 'audio',
        'b64': base64Encode(chunk),
        'is_final': isLast,
      }));
    }
  }

  // ==================== UI ====================
  Widget _loginScaffold() {
    return Scaffold(
      appBar: AppBar(title: const Text('原生语音 (P2)')),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.mic, size: 64, color: Colors.indigo),
              const SizedBox(height: 12),
              const Text('登录后使用原生语音',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
              const SizedBox(height: 8),
              Text(widget.serverUrl,
                  style: const TextStyle(fontSize: 12, color: Colors.grey)),
              const SizedBox(height: 24),
              TextField(
                controller: _userCtrl,
                decoration: const InputDecoration(
                    labelText: '用户名',
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.person)),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _passCtrl,
                obscureText: true,
                onSubmitted: (_) => _doLogin(),
                decoration: const InputDecoration(
                    labelText: '密码',
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.lock)),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: _loginBusy ? null : _doLogin,
                  child: _loginBusy
                      ? const SizedBox(
                          width: 20, height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2))
                      : const Text('登 录'),
                ),
              ),
              if (_loginErr.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Text(_loginErr,
                      style: const TextStyle(color: Colors.red, fontSize: 13)),
                ),
            ],
          ),
        ),
      ),
    );
  }

  void _addMsg(_Msg m) {
    setState(() {
      _msgs.add(m);
      if (_msgs.length > 200) _msgs.removeAt(0);
    });
    _scrollToBottom();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.jumpTo(_scrollCtrl.position.maxScrollExtent);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_loggedIn) return _loginScaffold();
    return Scaffold(
      appBar: AppBar(
        title: const Text('原生语音 (P2)'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Center(
              child: Text(_connState,
                  style: TextStyle(
                      fontSize: 12,
                      color: _connected ? Colors.green : Colors.red)),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // 状态行
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
            child: Row(
              children: [
                Expanded(
                    child: Text(
                        '$_statusLine${_sending ? ' · 等待回复…' : ''}',
                        style: const TextStyle(fontSize: 13),
                        overflow: TextOverflow.ellipsis)),
              ],
            ),
          ),
          // 消息区
          Expanded(
            child: ListView.builder(
              controller: _scrollCtrl,
              itemCount: _msgs.length,
              itemBuilder: (c, i) => _msgTile(_msgs[i]),
            ),
          ),
          // 控制区
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                // 模式按钮（按住/持续）
                ChoiceChip(
                  label: const Text('按住说话'),
                  selected: !_contMode,
                  onSelected: (_) => setState(() => _contMode = false),
                ),
                const SizedBox(width: 8),
                ChoiceChip(
                  label: const Text('持续对话'),
                  selected: _contMode,
                  onSelected: (_) => setState(() => _contMode = true),
                ),
                const Spacer(),
                // 说话按钮：按住（Listener onPointerDown/Up）或持续（toggle）
                _contMode
                    ? FilledButton(
                        onPressed: _listening ? _stopCapture : _startCapture,
                        child: Text(_listening ? '停止' : '开始'),
                      )
                    : Listener(
                        onPointerDown: (_) => _startCapture(),
                        onPointerUp: (_) => _stopCapture(finalize: true),
                        onPointerCancel: (_) => _stopCapture(finalize: true),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 28, vertical: 16),
                          decoration: BoxDecoration(
                            color: _listening ? Colors.redAccent : Colors.indigo,
                            borderRadius: BorderRadius.circular(30),
                          ),
                          child: Text(_listening ? '🎙 松开发送' : '🎤 按住说话',
                              style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 16,
                                  fontWeight: FontWeight.w600)),
                        ),
                      ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _msgTile(_Msg m) {
    final color = switch (m.kind) {
      'asr' => Colors.black87,
      'fast' => Colors.blue,
      'slow' => Colors.green.shade700,
      'err' => Colors.red,
      _ => Colors.black54,
    };
    final prefix = switch (m.kind) {
      'asr' => '🎤 ',
      'fast' => '⚡ ',
      'slow' => '🐢 ',
      'err' => '⚠️ ',
      _ => '',
    };
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 3),
      child: Align(
        alignment: m.kind == 'asr' ? Alignment.centerRight : Alignment.centerLeft,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.78),
          decoration: BoxDecoration(
            color: m.kind == 'asr'
                ? Colors.indigo.shade50
                : Colors.grey.shade100,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text('$prefix${m.text}',
              style: TextStyle(color: color, fontSize: 14)),
        ),
      ),
    );
  }
}

class _Msg {
  final String kind; // asr | fast | slow | err
  String text;
  final int? ms;
  final bool partial;
  _Msg({required this.kind, required this.text, this.ms, this.partial = false});
}
