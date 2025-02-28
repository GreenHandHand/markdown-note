---
cssclasses:
  - kanban
  - home
---

# Home

```dataviewjs
let today = new Date();
dv.paragraph(`今天是 [[${today.getFullYear()}-${(today.getMonth() + 1).toString().padStart(2, '0')}-${today.getDate().toString().padStart(2, '0')}]]。`)
```

```dataview
table without id "库中共有" +length(rows) + "个页面，总计约" + round(sum(rows.file.size)/10000,0) + "万字节（" + round((sum(rows.file.size)/1048576), 2) + "MB）" as "笔记库统计"
group by 12345
```

## Recent Files

```dataview
table file.mtime as "Last Edit", file.tags as "Tags"
from !#diary and !"Home" and !"template" and !"Readme" and !"Excalidraw"
sort file.mtime desc
limit 10
```

## Focus Counter

```dataview
table focusTime
FROM #diary 
WHERE focusTime
```

## Sleep Duration

```dataviewjs
const diary = await dv.pages('"日记"').filter(diary => (diary.getup && diary.getup != 'fill this') || (diary.sleep && diary.sleep != 'fill this')).sort(a => a.file.name);

const daysToShow = 15; // Change this to the number of days you want to display
const now = new Date();

// get the days to show
const pastDays = diary.filter(d => {
	const diaryDate = new Date(d.file.name);
	const diffTime = Math.abs(now - diaryDate);
	const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
	return diffDays <= daysToShow;
});

console.log(pastDays);

let getup = [], sleep = [], date = [], validEntries = [];
// Validate and process entries
for (let i = 1; i < pastDays.length; i++) {
    const prevDay = pastDays[i - 1];
    const currDay = pastDays[i];

    const prevSleep = prevDay.sleep && prevDay.sleep != 'fill this' ? parseTimeToMilliseconds(prevDay.sleep) : null;
    const currGetup = currDay.getup && currDay.getup != 'fill this' ? parseTimeToMilliseconds(currDay.getup) : null;
	// check date continuous
    const prevDate = new Date(prevDay.file.name);
	prevDate.setDate(prevDate.getDate() + 1);
	const prevDateTomorrow = prevDate.getDate();
    const currDate = new Date(currDay.file.name).getDate();

    // Ensure there's a valid transition and the dates are continuous
    if (prevSleep !== null && currGetup !== null && prevSleep <= currGetup && (currDate === prevDateTomorrow)) {
        validEntries.push(currDay);
        const diary_date = new Date(currDay.file.name);
        const getupTime = currGetup;
        const sleepTime = prevSleep;
        getup.push(getupTime);
        sleep.push(sleepTime);
        date.push(currDay.file.path);
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

let option = {
	height: "400px",
	width: "760px",
	backgroundColor: "transparent",
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
		var tar = params[0]; // Display the sleep duration
		return `${tar.name}<br/>Sleep Druation: ${formatTime(sleepDuration[tar.dataIndex])}<br/>Getup: ${formatTime(getup[tar.dataIndex])}<br/>Sleep: ${formatTime(sleep[tar.dataIndex])}`
		}
	},
	grid: {
		left: '5%',
		right: '3%',
		bottom: '3%',
		containLabel: true
	},
	xAxis: {
		type: 'category',
		splitLine: { show: false },
		data: date,
		axisLabel: {
			interval: 0,
			formatter: function (path, i) {
		       	const regex = /\d{4}-(\d{2})-(\d{2})/;
		       	const match = path.match(regex);
		   
		       	if (match) {
					if (i == date.length - 1){
						return `${match[1]}-${match[2]}\n(today)`;
					}else{
						return `${match[1]}-${match[2]}`;
					}
		       	}
		       	return null;
		   	}
		}
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
			name: 'Mean Getup Time',
			type: 'line',
			data: Array(date.length).fill(meanGetupTime),
			lineStyle: {
				color: '#1f78b4'
			}
		},
		{
			name: 'Mean Sleep Time',
			type: 'line',
			data: Array(date.length).fill(meanSleepTime),
			lineStyle: {
				color: '#33a02c'
			}
		},
		{
			name: 'Placeholder Getup',
			type: 'bar',
			stack: 'Getup',
			itemStyle: {
				borderColor: 'transparent',
				color: 'transparent'
			},
			data: sleep
		},
		{
			name: 'Getup Time',
			type: 'bar',
			stack: 'Getup',
			label: {
				show: true,
				position: 'inside',
				formatter: function (params) {
					return formatTime(params.value);
				}
			},
			data: sleepDuration,
			itemStyle: {
				color: function (params){
					// 最后一个颜色高亮，其余颜色为'#a6cee3'
					if(params.dataIndex == date.length - 1){
						return '#a6e3bb';
					}else{
						return '#a6cee3';
					}
				}
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

app.plugins.plugins['obsidian-echarts'].render(option, this.container);

this.container.style.display = "flex";
this.container.style.justifyContent = "center";
this.container.style.alignItems = "center";
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
  minWidth: 24px
  minHeight: 24px
cellStyleRules: []

```

```contributionGraph
title: 日记
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
