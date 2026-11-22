#!/usr/bin/env node
/**
 * 抓取 CME 30-Day Fed Funds Futures (ZQ) 日线收盘价，输出 CSV。
 *
 * 说明：CME 直连在部分网络环境不可用，这里通过 Barchart 页面上下文调用其
 * 内部 EOD 接口（需要页面 Cookie）。行情为第三方延迟数据，仅用于 P0 基准。
 *
 * 用法:
 *   node scripts/fetch_barchart.js --out data/market/zq_curve_latest.csv
 *   node scripts/fetch_barchart.js --start 2026-11 --months 14 --out /tmp/zq.csv
 *   node scripts/fetch_barchart.js --symbols ZQZ26,ZQF27,ZQG27
 *
 * 依赖:
 *   puppeteer-core（可用 PUPPETEER_CORE 指定路径）
 *   Chrome/Chromium（可用 CHROME_PATH 指定路径）
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
    '/tmp/browsercheck/node_modules/puppeteer-core',
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

function currentMonth() {
  const now = new Date(Date.now() + 8 * 3600 * 1000);
  return `${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, '0')}`;
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

async function main() {
  const start = arg('start', currentMonth());
  const months = Number(arg('months', '14'));
  const symbols = arg('symbols', '')
    ? arg('symbols').split(',').map((s) => s.trim().toUpperCase()).filter(Boolean)
    : generateSymbols(start, months);
  const out = arg('out', 'data/market/zq_curve_latest.csv');

  const puppeteer = loadPuppeteer();
  const browser = await puppeteer.launch({
    executablePath: findChrome(),
    headless: true,
    timeout: 90000,
    protocolTimeout: 180000,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--blink-settings=imagesEnabled=false'],
  });

  try {
    const page = await browser.newPage();
    await page.setUserAgent(
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
    );
    await page.goto('https://www.barchart.com/futures/quotes/ZQZ26', {
      waitUntil: 'networkidle2',
      timeout: 60000,
    });
    await new Promise((resolve) => setTimeout(resolve, 3000));

    const fetchSeries = async (sym, attempts = 3) => {
      for (let i = 0; i < attempts; i += 1) {
        try {
          return await page.evaluate(async (symbol) => {
            const url =
              '/proxies/timeseries/queryeod.ashx?symbol=' +
              symbol +
              '&data=daily&maxrecords=4&volume=contract&order=desc&dividends=false&backadjust=false';
            const response = await fetch(url, {
              credentials: 'include',
              headers: { Accept: 'text/plain,*/*' },
            });
            return { status: response.status, text: await response.text() };
          }, sym);
        } catch (err) {
          if (i === attempts - 1) throw err;
          await new Promise((resolve) => setTimeout(resolve, 2000));
        }
      }
      throw new Error(`fetch failed: ${sym}`);
    };

    const rows = [];
    for (const symbol of symbols) {
      const result = await fetchSeries(symbol);

      const first = result.text.trim().split('\n')[0] || '';
      if (result.status !== 200 || !first.startsWith(symbol + ',')) {
        console.error(`skip ${symbol}: status=${result.status} body=${first.slice(0, 80)}`);
        continue;
      }
      const cols = first.split(',');
      if (cols.length < 8) {
        console.error(`skip ${symbol}: unexpected columns=${cols.length}`);
        continue;
      }
      rows.push(cols.slice(0, 8));
      console.log(`${symbol} ${cols[1]} close=${cols[5]}`);
    }

    if (rows.length === 0) throw new Error('没有任何合约返回有效数据');

    const lines = ['symbol,as_of,open,high,low,close,volume,open_interest'].concat(
      rows.map((cols) => cols.join(','))
    );
    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, lines.join('\n') + '\n');
    console.log(`wrote ${rows.length} contracts -> ${out}`);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error(err.message || err);
  process.exit(1);
});
