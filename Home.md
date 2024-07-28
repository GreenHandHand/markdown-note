# Home

```dataviewjs
let today = new Date();
dv.paragraph(`今天是 [[${today.getFullYear()}-${(today.getMonth() + 1).toString().padStart(2, '0')}-${today.getDate()}]]。`)
```

```dataviewjs
const diary = dv.pages('"日记"').filter(diary => diary.getup);
let getup = [], sleep = [], date = [];

diary.forEach(d => {
  const diary_date = new Date(d.日期);
  const getupTime = parseTimeToMilliseconds(d.getup);
  const sleepTime = parseTimeToMilliseconds(d.sleep);

  getup.push(getupTime);
  sleep.push(sleepTime);
  date.push(`${diary_date.getMonth() + 1}-${diary_date.getDate()}`);
});

function parseTimeToMilliseconds(timeStr) {
  const [hours, minutes] = timeStr.split(':').map(Number);
  return hours * 3600000 + minutes * 60000; // 毫秒数
}

// 计算起床时间与睡眠时间之间的差值
const sleepDuration = getup.map((g, i) => ((g - sleep[i]) % (24 * 3600000)) || 24 * 3600000);

let option = {
  title: {
    text: '睡觉时间分布',
    subtext: 'Sleep Duration'
  },
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'shadow'
    },
    formatter: function (params) {
      var tar = params[1];
      return tar.name + '<br/>' + tar.seriesName + ' : ' + formatDuration(tar.value);
    }
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true
  },
  xAxis: {
    type: 'category',
    splitLine: { show: false },
    data: date,
  },
  yAxis: {
    type: 'value',
    name: 'Duration (hours)',
    axisLabel: {
      formatter: function (value) {
        return formatDuration(value);
      }
    }
  },
  series: [
    {
      name: 'Sleep Duration',
      type: 'bar',
      stack: 'Total',
      label: {
        show: true,
        position: 'inside'
      },
      data: sleepDuration
    }
  ]
};

// 添加一个格式化持续时间的辅助函数
function formatDuration(milliseconds) {
  const hours = Math.floor(milliseconds / 3600000);
  const minutes = Math.floor((milliseconds % 3600000) / 60000);
  return `${hours}h ${minutes}m`;
}

app.plugins.plugins['obsidian-echarts'].render(option, this.container);
```
## 足迹

```contributionGraph
title: 笔记
graphType: default
dateRangeValue: 180
dateRangeType: LATEST_DAYS
startOfWeek: 1
showCellRuleIndicators: true
titleStyle:
  textAlign: left
  fontSize: 15px
  fontWeight: normal
dataSource:
  type: PAGE
  value: ""
  dateField:
    type: FILE_MTIME
  filters: []
fillTheScreen: false
enableMainContainerShadow: false
mainContainerStyle:
  backgroundColor: "#ffffff00"
cellStyle:
  minWidth: 30px
  minHeight: 30px
cellStyleRules: []

```

```contributionGraph
title: 日记热力图
graphType: default
dateRangeValue: 180
dateRangeType: LATEST_DAYS
startOfWeek: 1
showCellRuleIndicators: true
titleStyle:
  textAlign: left
  fontSize: 15px
  fontWeight: normal
dataSource:
  type: PAGE
  value: "#diary"
  dateField:
    type: FILE_CTIME
fillTheScreen: false
enableMainContainerShadow: false
cellStyleRules:
  - id: Ocean_a
    color: "#8dd1e2"
    min: 1
    max: 2
  - id: Ocean_b
    color: "#63a1be"
    min: 2
    max: 3
  - id: Ocean_c
    color: "#376d93"
    min: 3
    max: 5
  - id: Ocean_d
    color: "#012f60"
    min: 5
    max: 9999
cellStyle:
  minWidth: 25px
  minHeight: 25px
mainContainerStyle:
  backgroundColor: "#ffffff00"

```
