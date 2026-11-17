# nccl-tests 涓?Keysight KAI config/report 鎸囨爣瀛楁瀵规瘮

鏃ユ湡锛?026-10-25

## 1. 鑼冨洿涓庡師濮嬫潵婧?
鏈姤鍛婂彧浣跨敤鏈湴涓€鎵嬫枃浠讹細

- NCCL-tests 婧愮爜锛歚E:\dpi\nccl-tests-master\README.md`銆乣doc\PERFORMANCE.md`銆?- KAI 鏍锋湰閰嶇疆锛歚D:\宸ュ叿寮€鍙慭鍚岃\KAI鎴浘鍜屽綍灞廫config&results\trial_result_0e65a20d-3caa-4d31-a795-f2044fb9348d-1\trial_results_2026-10-21_07_59_12\result\2026-10-21\07-33-43_0e65a20d-3caa-4d31-a795-f2044fb9348d\trial.yaml`銆?- KAI 鏍锋湰缁撴灉锛氬悓鐩綍 `csv\summary.csv`銆乣iteration_metric.csv`銆乣interval_metrics.csv`銆乣port_metrics.csv`銆乣qp_metrics.csv`銆乣flow_metrics.csv` 鍙婂搴?Feather/meta 鏂囦欢銆?
NCCL-tests 鏈韩涓嶆槸涓€涓浐瀹?YAML 閰嶇疆鏍煎紡鐨勬祴璇曞钩鍙般€傚畠閫氳繃鍛戒护琛屽弬鏁板拰 NCCL 鐜鍙橀噺閰嶇疆锛屾枃鏈〃鏍兼垨 `-J` JSON 杈撳嚭鎶ュ憡銆侹AI 鍒欎繚瀛樹簡鎷撴墤銆佷华琛ㄧ鍙ｃ€丷oCE銆佹嫢濉炴帶鍒躲€侀泦鍚堥€氫俊绠楁硶銆佹暟鎹妯″拰鏁呴殰娉ㄥ叆绛夊畬鏁?trial 閰嶇疆锛屽苟鎸夊涓娴嬪眰娆¤惤鐩樼粨鏋溿€?
## 2. NCCL-tests 鐨勯厤缃瓧娈?
渚濇嵁 README 鐨勫懡浠よ璇存槑锛屼富瑕侀厤缃淮搴﹀涓嬶細

| 閰嶇疆缁村害 | NCCL-tests 瀛楁/鍙傛暟 | 鍚箟 |
|---|---|---|
| 娴嬭瘯绋嬪簭 | `all_reduce_perf`銆乣all_gather_perf`銆乣broadcast_perf` 绛?| 閫夋嫨闆嗗悎閫氫俊鎿嶄綔 |
| 鏁版嵁鑼冨洿 | `-b/--minbytes`銆乣-e/--maxbytes`銆乣-i/--stepbytes`銆乣-f/--stepfactor` | 鏁版嵁閲忚捣鐐广€佺粓鐐瑰拰姝ヨ繘 |
| 鏁版嵁绫诲瀷 | `-d/--datatype` | Float銆丠alf銆両nt 绛?NCCL 鏁版嵁绫诲瀷 |
| 鏍?rank | `-r/--root` | Broadcast/Reduce 绛夐渶瑕?root 鐨勬搷浣?|
| rank/GPU 骞惰搴?| `-g/--ngpus`銆丮PI 杩涚▼鏁般€佺嚎绋嬫暟 | 鍙備笌娴嬭瘯鐨?GPU/rank 鏁帮紱鎬?rank 鏄繘绋嬫暟脳绾跨▼鏁懊楁瘡绾跨▼ GPU 鏁?|
| 杩唬 | `-n/--iters`銆乣-w/--warmup_iters`銆乣-m/--agg_iters`銆乣-N/--run_cycles` | 姝ｅ紡杩唬銆侀鐑€佽仛鍚堛€佸惊鐜鏁?|
| 姝ｇ‘鎬?| `-c/--check` | 缁撴灉鏍￠獙杩唬娆℃暟 |
| 杩愯琛屼负 | `-p/--parallel_init`銆乣-C/--report_cputime`銆乣-G/--cudagraph`銆乣-T/--timeout` | 鍒濆鍖栥€佽鏃跺彛寰勩€丆UDA graph銆佽秴鏃?|
| 杈撳嚭 | `-S/--report_timestamps`銆乣-J/--output_file`銆乣-U/--tuning` | 鏃堕棿鎴炽€丣SON 鏂囦欢銆丯CCL tuning 淇℃伅 |
| 閫氫俊閫夋嫨 | `NCCL_ALGO`銆乣NCCL_PROTO`銆乣NCCL_TESTS_SPLIT` 绛夌幆澧冨彉閲?| 绠楁硶銆佸崗璁拰 rank 鍒嗙粍锛汵CCL 杩愯鏃跺彉閲忚繕褰卞搷瀹為檯璺緞 |

NCCL-tests 鐨勯厤缃噸鐐规槸鈥滆繍琛屼竴娆￠泦鍚堥€氫俊鎬ц兘娴嬭瘯闇€瑕佸摢浜涘弬鏁扳€濓紝涓嶆弿杩颁氦鎹㈡満銆佺墿鐞嗙鍙ｃ€丯IC 鐨勯摼璺眰鍙傛暟锛屼篃娌℃湁 KAI 涓偅绉嶇嫭绔嬬殑浠〃缁戝畾妯″瀷銆?
## 3. NCCL-tests 鐨勬姤鍛婂瓧娈?
鏍囧噯鏂囨湰杈撳嚭鐨勬牳蹇冨垪涓猴細

| NCCL-tests 瀛楁 | 鍗曚綅/鍙ｅ緞 | 璇存槑 |
|---|---|---|
| `size` | B | 鏈疆浼犺緭/鎿嶄綔鏁版嵁瑙勬ā |
| `count` | elements | 姣?rank 鐨勫厓绱犳暟閲?|
| `type` | - | 鏁版嵁绫诲瀷 |
| `redop` | - | 瑙勭害鎿嶄綔锛屽 Sum |
| `root` | - | root rank锛涙棤 root 鎿嶄綔閫氬父涓?`-` |
| `time` | us 鎴?ms锛屾寜杈撳嚭琛ㄥご | 骞冲潎鎿嶄綔鏃堕棿锛沗doc/PERFORMANCE.md` 瑙ｉ噴鍏朵负骞冲潎 operation time |
| `algbw` | GB/s | 绠楁硶甯﹀锛宍S/t` |
| `busbw` | GB/s | 鎸夐泦鍚堥€氫俊娴侀噺鎶樼畻鐨勬€荤嚎甯﹀锛涗笉鍚?collective 鐨勬崲绠楃郴鏁颁笉鍚?|
| `#wrong` | count | 姝ｇ‘鎬ч敊璇暟锛涘簲涓?0 |

杈撳嚭杩樺尯鍒?`out-of-place` 鍜?`in-place`銆俁EADME 璇存槑鍙互浣跨敤 `-J` 鐢熸垚 JSON锛涘洜姝や骇鍝佸寲鏃跺簲浼樺厛閲囬泦 JSON锛岃€屼笉鏄В鏋愬榻愭枃鏈€?
`algbw` 鍜?KAI 鐨?`Algbw (GB/s)` 鍙互鐩存帴瀵归綈锛沗busbw` 鍜?KAI 鐨?`Busbw (GB/s)` 涔熷彲浠ョ洿鎺ュ榻愶紝浣嗗墠鎻愭槸 collective銆乺ank 鏁般€佹暟鎹妯″彛寰勪竴鑷淬€侼CCL 鏂囨。鏄庣‘璇存槑锛宍busbw` 鏄负浜嗗弽鏄犵‖浠跺埄鐢ㄦ晥鐜囪€屽紩鍏ョ殑鎸囨爣锛屼笉鑳芥妸瀹冨綋鎴愬師濮嬬嚎閫熴€?
## 4. KAI 閰嶇疆瀛楁锛坱rial.yaml锛?
KAI 鐨勯厤缃瘮 NCCL-tests 鏇存帴杩戔€滅綉缁滄祴璇曚华 + 琚祴绯荤粺鈥濈殑璇曢獙鎻忚堪锛屼富瑕佸瓧娈靛涓嬶細

| KAI 閰嶇疆鍖?| 鍏抽敭瀛楁 | 瀵瑰簲鍚箟 |
|---|---|---|
| `platform` | `PLATFORM_KEYS_HW` | 娴嬭瘯骞冲彴/浠〃绫诲瀷 |
| `infrastructure.deviceInstances` | `generic`銆乣blackbox_switch`銆乣count` | 涓绘満銆佷氦鎹㈡満鍜屽疄渚嬫暟閲?|
| `physicalBindings` | `chassisLocation.address`銆乣port`銆乣layer1.speedMode` | 浠〃鏈虹 IP銆佺墿鐞嗙鍙ｃ€?00G NRZ/RS-FEC 閾捐矾妯″紡 |
| `nicSettings` | MAC銆両Pv4銆丮TU銆乂LAN銆丵oS | 琚祴 NIC 鐨勪簩灞?涓夊眰璁剧疆 |
| `roceTransportSettings` | `ackDscp`銆乣dataDscp`銆乣nackDscp` | RoCE 娴侀噺 DSCP |
| `congestionControl` | ECN銆丳FC銆丏CQCN 鍙傛暟 | 鎷ュ鎺у埗涓庢棤鎹熺綉缁滆缃?|
| `rankBindings` | rankId銆丯PU銆丯IC 寮曠敤 | rank 鍒?NIC/NPU 鐨勬嫇鎵戠粦瀹?|
| `kccb.benchmark` | `collectiveAlgorithm.system`銆乣datasize.start/end/step`銆乣iterations`銆乣iterationAppendDelay` | 闆嗗悎閫氫俊绠楁硶銆佹暟鎹寖鍥淬€佽凯浠ｅ拰棰濆寤惰繜 |
| `rocev2` | QP 鏁般€佹秷鎭ぇ灏忋€侀噸浼犮€丷NR銆乿erb | RoCEv2 浼犺緭琛屼负 |
| `impairments` | packetDrop銆乸acketReorder | 涓㈠寘銆佷贡搴忕瓑鏁呴殰娉ㄥ叆 |
| `infrastructureProfile` | host 鏁般€丯PU 鏁般€佸甫瀹?| 娴嬭瘯鎷撴墤鑳藉姏妯″瀷 |

鏍锋湰閰嶇疆鏄?2 涓?host銆佹瘡涓?1 涓?NPU/NIC锛屼氦鎹㈡満绔彛涓?`10.11.58.151:1.1` 鍜?`3.1`锛?00G 閾捐矾锛岄泦鍚堢畻娉曚负 `ALGO_ALL_GATHER_RING`锛屾暟鎹粠 128 MiB 鍒?4 GiB銆佹寜 2 鍊嶉€掑锛岃凯浠ｆ暟涓?1锛涘悓鏃跺惎鐢ㄤ簡 PFC锛岄厤缃簡 packet drop/reorder impairment銆?
## 5. KAI 鎶ュ憡瀛楁

### 5.1 summary.csv锛氫富缁撴灉灞?
瀛楁锛歚Total Bytes Tx`銆乣Total Bytes Rx`銆乣Iterations`銆乣Collective Time (ms)`銆乣Algbw (GB/s)`銆乣Busbw (GB/s)`銆乣Ideal (%)`銆乣PFC Rx`銆乣ECN CE Rx`銆乣Frames ReTx`銆乣Total Time (s)`銆乣Duty Cycle (%)`銆乣CNP Tx`銆乣CNP Rx`銆乣Data size (B)`銆乣Collective`銆?
杩欐槸鏈€閫傚悎鍜?NCCL-tests 涓绘姤鍛婄洿鎺ュ姣旂殑涓€灞傦細

- `Data size (B)` 瀵?`size`锛?- `Collective Time (ms)` 瀵?`time`锛?- `Algbw` 瀵?`algbw`锛?- `Busbw` 瀵?`busbw`锛?- `Iterations` 瀵?`-n`/`kccb.benchmark.iterations`锛?- `Collective` 瀵规祴璇曟搷浣滃拰搴忓彿锛?- `Ideal (%)` 鏄?KAI 棰濆鎻愪緵鐨勭浉瀵圭悊鎯冲甫瀹藉埄鐢ㄧ巼锛汵CCL-tests 鏍囧噯鍒楁病鏈夊悓鍚嶅瓧娈碉紱
- PFC銆丒CN銆侀噸浼犮€丆NP 鏄綉缁滃仴搴锋寚鏍囷紝NCCL-tests 鏍囧噯杈撳嚭娌℃湁瀵瑰簲鍒椼€?
### 5.2 iteration_metric.csv锛氭寜杩唬鐨勬椂搴忓拰鍒嗗竷

瀛楁鍖呮嫭 `Collective Start/End/Completion Time`銆乣Algbw`銆乣Busbw`銆乣Ideal` 浠ュ強 `Min/P50/P95/Max FCT`銆傚畠鑳藉琛ㄨ揪鍗曡疆/澶氳疆鎶栧姩鍜?FCT 鍒嗗竷锛汵CCL-tests 鏍囧噯鏂囨湰閫氬父鍙姤鍛婅仛鍚堝悗鐨勫钩鍧囨椂闂达紝涓嶈兘鐩存帴鎻愪緵 P50/P95/Max銆?
### 5.3 interval_metrics.csv锛氳仛鍚堢粺璁″眰

瀛楁鍖呮嫭 `Min/Max/Avg Collective Time`銆乣Min/Max/Avg Algbw`銆乣Min/Max/Avg Busbw`銆乣Min/Max/Avg Ideal`锛屼互鍙?PFC銆丒CN銆侀噸浼犮€佹€绘椂闂村拰 Duty Cycle銆傚畠鏄€傚悎鐢昏秼鍔垮拰灞曠ず鍖洪棿缁熻鐨勭粨鏋滃眰銆?
### 5.4 port_metrics.csv锛氱鍙?閾捐矾灞?
瀛楁鍖呮嫭鏈虹鍦板潃/绔彛銆丷anks銆丩ink State銆丩ink Speed銆丗rames/Bytes Tx/Rx銆丆RC Errors銆丳ause Frames Rx銆丷oCEv2 QPs Configured/Ready銆佹暟鎹抚銆丄CK/NACK銆丗rame Sequence Errors銆丒CN-CE銆丆NP銆侀噸浼犮€傚畠鎶婃€ц兘缁撴灉鍜岀墿鐞嗛摼璺?鍗忚璁℃暟鍣ㄥ叧鑱旇捣鏉ワ紝NCCL-tests 娌℃湁绛変环鐨勬爣鍑嗙粨鏋滃瓧娈点€?
### 5.5 qp_metrics.csv锛歈P/璺緞灞?
瀛楁鍖呮嫭鏁版嵁瑙勬ā銆乧ollective銆佹簮/鐩殑 QP銆佹簮/鐩殑 rank/NIC銆佸彂閫?鎺ユ敹/澶辫触娑堟伅銆乀x/Rx bytes銆佸抚璁℃暟銆佸簭鍙烽敊璇€侀噸浼犮€丄CK/NACK銆丒CN/CNP銆佹渶灏?骞冲潎/鏈€澶?latency銆佷贡搴忓拰閲嶅鍖呫€侼CCL-tests 鐨勬爣鍑嗚緭鍑轰笉鏆撮湶 QP 绾ц娴嬨€?
### 5.6 flow_metrics.csv锛氭祦/鏁版嵁鍧楀眰

瀛楁鍖呮嫭 Data size銆丆ollective銆両teration銆丏ata Chunk Transfer ID銆丵P ID銆丆hannel ID銆丅uffer Offset銆丆hunk Count銆佹簮/鐩殑 rank/NIC銆丼tart/End/Completion time銆乀ransmit Type銆丒mulated Tx Component銆丗low Size銆傚畠鑳借拷韪叿浣撴暟鎹潡鐨勮矾寰勫拰瀹屾垚鏃堕棿锛孨CCL-tests 娌℃湁绛変环灞傘€?
## 6. 鐩存帴鍙瘮涓庝笉鍙洿鎺ユ瘮瀛楁

| 姣旇緝缁撹 | 瀛楁 |
|---|---|
| 鍙洿鎺ュ榻?| 鏁版嵁瑙勬ā锛歂CCL `size` 鈫?KAI `Data size (B)`锛涙椂闂达細`time` 鈫?`Collective Time (ms)`锛涚畻娉曞甫瀹斤細`algbw` 鈫?`Algbw`锛涙€荤嚎甯﹀锛歚busbw` 鈫?`Busbw`锛涙搷浣滐細娴嬭瘯绋嬪簭/collective 鈫?`Collective`锛涜凯浠ｆ鏁帮細`-n` 鈫?`Iterations` |
| 鍙湪缁熶竴鍙ｅ緞鍚庡榻?| NCCL `count` 闇€瑕佺粨鍚?`type` 鐨勫厓绱犲瓧鑺傛暟鎹㈢畻鎴?bytes锛汯AI 涓昏鐩存帴璁板綍 bytes銆侼CCL `root`/`redop` 闇€瑕佹斁鍏ユ搷浣滃厓鏁版嵁锛屼笌 KAI 鐨?collective algorithm 涓€璧疯В閲娿€侼CCL `time` 鐨勮〃澶村崟浣嶅彲鑳芥槸 us锛孠AI summary 鏄庣‘涓?ms锛岄渶缁熶竴鍗曚綅銆?|
| KAI 鐙湁 | `Ideal (%)`銆丳FC/ECN/CNP銆丗rames ReTx銆丏uty Cycle銆佺鍙ｉ摼璺€丆RC銆丳ause銆丵P銆乫low銆丗CT 鍒嗕綅鏁般€佹晠闅滄敞鍏ョ粨鏋?|
| NCCL-tests 鐙湁鎴栨爣鍑嗗垪鐙湁 | `count`銆乣type`銆乣redop`銆乣root`銆乣#wrong`銆乮n-place/out-of-place銆丯CCL tuning 淇℃伅 |

## 7. 鍏抽敭鍙ｅ緞椋庨櫓

1. **collective 绠楁硶涓嶅悓浼氭敼鍙?busbw 鐨勫惈涔夈€?* NCCL `busbw` 浣跨敤鎸?collective 瀹氫箟鐨勬崲绠楀洜瀛愶紱KAI 鏍锋湰鏄?`ALL_GATHER`锛屾瘮杈冩椂蹇呴』浣跨敤鐩稿悓 collective 鍜岀浉鍚?rank 鏁般€?2. **鏃堕棿鍗曚綅涓嶅悓銆?* NCCL 杈撳嚭甯歌琛ㄥご涓?`us`锛孠AI summary/iteration 鏄庣‘鏄?`ms`銆傜粺涓€妯″瀷寤鸿鍐呴儴浣跨敤绾崇鎴栧井绉掞紝灞曠ず灞傚啀杞崲銆?3. **鏁版嵁瑙勬ā涓嶄竴瀹氭槸鍚屼竴涓€昏緫閲忋€?* NCCL 鐨?`size`/`count` 涓?KAI 鐨?`Data size (B)` 闇€瑕佺‘璁ゆ槸 per-rank銆乻end buffer 杩樻槸 collective 鎬婚噺锛涗笉鑳戒粎鍑垪鍚嶅悎骞躲€?4. **閿欒涓庣綉缁滆鏁板櫒涓嶈兘娣蜂负涓€璋堛€?* NCCL `#wrong` 鏄粨鏋滄纭€ч敊璇紱KAI 鐨?`CRC Errors`銆丯ACK銆侀噸浼犮€丒CN 鏄綉缁?鍗忚浜嬩欢锛屼笉绛変环浜?`#wrong`銆?5. **NCCL-tests 鐨勫钩鍧囧€间笌 KAI 鍒嗗竷鍊煎眰绾т笉鍚屻€?* KAI 鐨?FCT銆乵in/p50/p95/max 闇€瑕佸崟鐙缓妯★紱涓嶈兘瑕嗙洊 NCCL 鐨勫钩鍧?`time`銆?
## 8. 寤鸿鍛堢幇鐨勭粺涓€鎶ュ憡瀛楁

鏍规嵁褰撳墠瑕佹眰锛岀晫闈㈠彧鍛堢幇 nccl-tests 瀹為檯鑳藉浜х敓銆佹垨鑳藉浠庡叾瀹樻柟 JSON report 绋冲畾璇诲彇鐨勫瓧娈点€侹AI 鐙湁鑰?nccl-tests 娌℃湁瀵瑰簲鏉ユ簮鐨勬寚鏍囦笉杩涘叆 NCCL 鎶ュ憡椤碉紝涔熶笉浠?`unavailable` 鎴?0 鐨勫舰寮忎吉閫犲睍绀恒€俙r`n
**娴嬭瘯鍏冩暟鎹眰**锛氳繍琛屾椂闂淬€佷富鏈?杩涚▼/rank/GPU 鏁般€乧ollective銆丯CCL algorithm/protocol锛堣嫢鎶ュ憡鎻愪緵锛夈€乨atatype銆乺oot銆乺edop銆乮n-place/out-of-place銆亀armup銆佹寮忚凯浠ｆ鏁般€佹暟鎹妯¤捣姝㈠拰姝ヨ繘銆丯CCL/CUDA/椹卞姩鐗堟湰銆俙r`n
**涓绘€ц兘灞?*锛歚size`銆乣count`銆乣time`銆乣algbw`銆乣busbw`銆乣#wrong`锛屼互鍙?JSON report 鍙彁渚涚殑瀹為檯杩唬缁熻锛歚min_us`銆乣max_us`銆乣avg_us`銆乣p50_us`銆乣p95_us`銆乣p99_us`銆乣stdev_us`銆乣cv_pct`銆俙r`n
**鏍￠獙涓庢眹鎬诲眰**锛歰ut-of-bounds 妫€鏌ユ暟閲忓拰鐘舵€併€佸钩鍧?bus bandwidth锛堝鏋滆娆℃姤鍛婃彁渚涳級銆佹祴璇曟垚鍔?澶辫触鐘舵€併€俙r`n
浠ヤ笅瀛楁涓嶇撼鍏?NCCL-tests 鍘熷瀷鐨勫憟鐜拌寖鍥达細`Ideal (%)`銆丳FC銆丒CN銆丆NP銆丗rames ReTx銆丏uty Cycle銆丆RC銆丳ause銆佺鍙ｉ摼璺姸鎬併€丵P 绾ф寚鏍囥€乫low/FCT 鍒嗕綅鏁板拰 KAI 鏁呴殰娉ㄥ叆缁撴灉銆傚畠浠睘浜?KAI 缃戠粶浠〃鎴栦豢鐪熷櫒瑙傛祴锛屽綋鍓?nccl-tests 娌℃湁鍚岀瓑鏁版嵁婧愩€俙r`n
## 9. 鏂囦欢绾ц瘉鎹储寮?
- NCCL CLI/config锛歚E:\dpi\nccl-tests-master\README.md:37-106`
- NCCL 鎸囨爣瀹氫箟锛歚E:\dpi\nccl-tests-master\doc\PERFORMANCE.md:3-27`锛屼互鍙婅鏂囦欢涓悇 collective 鐨?busbw 鍏紡銆?- KAI 閰嶇疆锛歚...\trial.yaml` 鐨?`binding`銆乣kccb.benchmark`銆乣rocev2`銆乣impairments`銆乣infrastructureProfile`銆?- KAI 涓荤粨鏋滐細`...\csv\summary.csv`銆?- KAI 杩唬缁熻锛歚...\csv\iteration_metric.csv`銆乣...\csv\interval_metrics.csv`銆?- KAI 閾捐矾/QP/娴佺粨鏋滐細`...\csv\port_metrics.csv`銆乣...\csv\qp_metrics.csv`銆乣...\csv\flow_metrics.csv`銆?
