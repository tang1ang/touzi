#!/usr/bin/env node
/**
 * 抓取 CME 30-Day Fed Funds Futures (ZQ) 历史日线，输出长表 CSV。
 *
 * 通过 Barchart 页面上下文调用其内部 EOD 接口（需要页面 Cookie），
 * 行情为第三方延迟数据，仅用于回测。
 *
 * 用法:
 *   node scripts/fetch_barchart_history.js --start 2023-11 --months 37 --out data/backtest/zq_history.csv
 *   node scripts/fetch_barchart_history.js --symbols ZQZ23,ZQF24 --out /tmp/zq.csv
 */

const fs = require('fs');
const path = require('path');

const MONTH_CODES = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'];

function arg(name, fallback) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] || fallback;
}

function loadPuppeteer() {
  const candidates = [
    process.env.PUPPETEER_CORE,
    'puppeteer-core',
    '/tmp/node_modules/puppeteer-core',
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      return require(candidate);
    } catch (err) {
      /* try next */
    }
  }
  throw new Error('找不到 puppeteer-core：请 npm i puppeteer-core 或设置 PUPPETEER_CORE');
}

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    '/root/.cache/puppeteer/chrome/linux-155.0.8059.39/chrome-linux64/chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  throw new Error('找不到 Chrome：请设置 CHROME_PATH');
}

function generateSymbols(start, months) {
  const [year0, month0] = start.split('-').map(Number);
  if (!year0 || !month0) throw new Error(`非法起始月份: ${start}`);
  let year = year0;
  let month = month0;
  const symbols = [];
  for (let i = 0; i < months; i += 1) {
    symbols.push(`ZQ${MONTH_CODES[month - 1]}${String(year % 100).padStart(2, '0')}`);
    month += 1;
    if (month > 12) {
      month = 1;
      year += 1;
    }
  }
  return symbols;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  const start = arg('start', '2023-11');
  const months = Number(arg('months', '37'));
  const from = arg('from', '2023-10-01');
  const maxrecords = Number(arg('maxrecords', '1000'));
  const symbols = arg('symbols', '')
    ? arg('symbols').split(',').map((s) => s.trim().toUpperCase()).filter(Boolean)
    : generateSymbols(start, months);
  const out = arg('out', 'data/backtest/zq_history.csv');

  const puppeteer = loadPuppeteer();
  const browser = await puppeteer.launch({
    executablePath: findChrome(),
    headless: true,
    timeout: 90000,
    protocolTimeout: 300000,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--blink-settings=imagesEnabled=false'],
  });

  try {
    const page = await browser.newPage();
    await page.setUserAgent(
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
    );
    await page.goto('https://www.barchart.com/futures/quotes/ZQZ26', {
      waitUntil: 'networkidle2',
      timeout: 90000,
    });
    await sleep(5000);

    const fetchSeries = async (sym, attempts = 3) => {
      for (let i = 0; i < attempts; i += 1) {
        try {
          return await page.evaluate(
            async (symbol, records) => {
              const url =
                '/proxies/timeseries/queryeod.ashx?symbol=' +
                symbol +
                '&data=daily&maxrecords=' +
                records +
                '&volume=contract&order=desc&dividends=false&backadjust=false';
              const response = await fetch(url, {
                credentials: 'include',
                headers: { Accept: 'text/plain,*/*' },
              });
              return { status: response.status, text: await response.text() };
            },
            sym,
            maxrecords
          );
        } catch (err) {
          if (i === attempts - 1) throw err;
          await sleep(2000);
        }
      }
      throw new Error(`fetch failed: ${sym}`);
    };

    const rows = [];
    let ok = 0;
    for (const symbol of symbols) {
      const result = await fetchSeries(symbol);
      const lines = result.text.trim().split('\n').filter(Boolean);
      const first = lines[0] || '';
      if (result.status !== 200 || !first.startsWith(symbol + ',')) {
        console.error(`skip ${symbol}: status=${result.status} body=${first.slice(0, 80)}`);
        continue;
      }
      let count = 0;
      for (const line of lines) {
        const cols = line.split(',');
        if (cols.length < 8 || cols[0] !== symbol) continue;
        if (cols[1] < from) continue;
        rows.push(cols.slice(0, 8));
        count += 1;
      }
      ok += 1;
      console.log(`${symbol}: ${count} rows`);
      await sleep(300);
    }

    if (rows.length === 0) throw new Error('没有任何合约返回有效数据');

    const header = 'symbol,as_of,open,high,low,close,volume,open_interest';
    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, [header].concat(rows).join('\n') + '\n');
    console.log(`wrote ${ok} contracts / ${rows.length} rows -> ${out}`);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error(err.message || err);
  process.exit(1);
});
