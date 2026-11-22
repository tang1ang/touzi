/* Fed Policy Radar — charts */
(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  var el = document.getElementById('chart-slo');
  if (!el) return;

  var chart = echarts.init(el, null, { renderer: 'svg' });

  var rows = [
    { name: '官方数据入库', value: 30, color: accent },
    { name: '行情快照入库', value: 60, color: accent },
    { name: '特征更新', value: 60, color: accent },
    { name: '概率重算与推送', value: 120, color: accent },
    { name: '告警推送', value: 150, color: accent },
    { name: '文本评分', value: 900, color: accent2 }
  ];

  function fmt(v) {
    if (v < 60) return v + ' 秒';
    if (v < 3600) {
      var m = v / 60;
      return (m % 1 === 0 ? m : m.toFixed(1)) + ' 分钟';
    }
    return (v / 3600).toFixed(1) + ' 小时';
  }

  chart.setOption({
    animation: false,
    grid: { left: 118, right: 66, top: 18, bottom: 34 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      appendToBody: true,
      backgroundColor: bg2,
      borderColor: rule,
      textStyle: { color: ink, fontSize: 12 },
      formatter: function (params) {
        var p = params[0];
        return p.name + '<br/>目标：' + fmt(p.value);
      }
    },
    xAxis: {
      type: 'log',
      min: 20,
      max: 1000,
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } },
      axisLabel: {
        color: muted,
        fontSize: 10,
        formatter: function (v) { return v + 's'; }
      }
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: rows.map(function (r) { return r.name; }),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: ink, fontSize: 11.5 }
    },
    series: [{
      type: 'bar',
      barWidth: 13,
      data: rows.map(function (r) {
        return {
          value: r.value,
          itemStyle: { color: r.color, borderRadius: [0, 2, 2, 0] }
        };
      }),
      label: {
        show: true,
        position: 'right',
        color: muted,
        fontSize: 10.5,
        formatter: function (p) { return fmt(p.value); }
      }
    }]
  });

  window.addEventListener('resize', function () { chart.resize(); });
})();
