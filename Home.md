# Home

```dataviewjs
let today = new Date();
dv.paragraph(`今天是 [[${today.getFullYear()}-${(today.getMonth() + 1).toString().padStart(2, '0')}-${today.getDate()}]]。`)
```
```dataviewjs
const diary = dv.pages('"日记"').filter(diary => diary.getup && diary.getup != 'fill this' && diary.sleep && diary.sleep != 'fill this').sort(a => a.日期);

console.log(diary);

// Filter to include only the past few days
const daysToShow = 10; // Change this to the number of days you want to display
const now = new Date();
const pastDays = diary.filter(d => {
  const diaryDate = new Date(d.日期);
  const diffTime = Math.abs(now - diaryDate);
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return diffDays <= daysToShow;
});

let getup = [], sleep = [], date = [], validEntries = [];

// Validate and process entries
for (let i = 1; i < pastDays.length; i++) {
  const prevDay = pastDays[i - 1];
  const currDay = pastDays[i];
  const prevSleep = parseTimeToMilliseconds(prevDay.sleep);
  const currGetup = parseTimeToMilliseconds(currDay.getup);

  // Ensure the sleep time of the previous day and the getup time of the current day are valid
  if (prevSleep <= currGetup) {
    validEntries.push(currDay);
    const diary_date = new Date(currDay.日期);
    const getupTime = currGetup;
    const sleepTime = parseTimeToMilliseconds(currDay.sleep);

    getup.push(getupTime);
    sleep.push(sleepTime);
    date.push(`${diary_date.getMonth() + 1}-${diary_date.getDate()}`);
  }
}

function parseTimeToMilliseconds(timeStr) {
  const [hours, minutes] = timeStr.split(':').map(Number);
  return hours * 3600000 + minutes * 60000; // 毫秒数
}

// Calculate sleep duration
const sleepDuration = getup.map((g, i) => {
  const duration = g - sleep[i];
  return duration >= 0 ? duration : duration + 24 * 3600000;
});

// Calculate means for getup and sleep times
const meanGetupTime = Math.round(getup.reduce((a, b) => a + b, 0) / getup.length);
const meanSleepTime = Math.round(sleep.reduce((a, b) => a + b, 0) / sleep.length);

// Adjust the chart's dimensions
const chartWidth = window.innerWidth - 20; // Leave some margin
const chartHeight = window.innerHeight - 100; // Leave some margin for titles, etc.

let option = {
  title: {
    text: '睡觉时间分布',
    subtext: 'Sleep Duration',
    left: 'center'
  },
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'shadow'
    },
    formatter: function (params) {
      var tar = params[1];
      return tar.name + '<br/>' + tar.seriesName + ' : ' + formatTime(tar.value);
    }
  },
  legend: {
    data: ['Sleep Duration', 'Mean Getup Time', 'Mean Sleep Time'],
    top: 'bottom'
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
    name: 'Time (HH:MM)',
    axisLabel: {
      formatter: function (value) {
        return formatTime(value);
      }
    }
  },
  series: [
    {
      name: 'Sleep Duration',
      type: 'bar',
      stack: 'Total',
      barGap: '10%', // Add margin between bars
      label: {
        show: true,
        position: 'inside',
        formatter: function (params) {
          return formatTime(params.value);
        }
      },
      data: sleepDuration,
      itemStyle: {
        color: '#66c2a5'
      }
    },
    {
      name: 'Mean Getup Time',
      type: 'line',
      data: Array(date.length).fill(meanGetupTime),
      lineStyle: {
        color: '#fc8d62'
      }
    },
    {
      name: 'Mean Sleep Time',
      type: 'line',
      data: Array(date.length).fill(meanSleepTime),
      lineStyle: {
        color: '#8da0cb'
      }
    },
    {
      name: 'Area Between Means',
      type: 'line',
      data: Array(date.length).fill(meanGetupTime),
      markArea: {
        silent: true,
        itemStyle: {
          color: 'rgba(128, 128, 128, 0.1)'
        },
        data: [[
          {
            yAxis: meanSleepTime
          },
          {
            yAxis: meanGetupTime
          }
        ]]
      }
    }
  ]
};

// Format time in HH:MM
function formatTime(milliseconds) {
  const totalMinutes = milliseconds / 60000;
  const hours = Math.floor(totalMinutes / 60);
  const minutes = Math.floor(totalMinutes % 60);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
}

app.plugins.plugins['obsidian-echarts'].render(option, this.container, {width: chartWidth, height: chartHeight});

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
