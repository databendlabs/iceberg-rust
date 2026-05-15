// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;

use clap::{Parser, ValueEnum};
use datafusion::arrow::csv::WriterBuilder as CsvWriterBuilder;
use datafusion::arrow::json::{ArrayWriter, LineDelimitedWriter};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::dataframe::DataFrame;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use iceberg_playground::{ICEBERG_PLAYGROUND_VERSION, IcebergCatalogList};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PrintFormat {
    Csv,
    Tsv,
    Table,
    Json,
    Ndjson,
    Automatic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaxRows {
    Limited(usize),
    Unlimited,
}

impl FromStr for MaxRows {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "inf" => Ok(Self::Unlimited),
            _ => s
                .parse::<usize>()
                .map(Self::Limited)
                .map_err(|e| e.to_string()),
        }
    }
}

#[derive(Debug, Parser, PartialEq)]
#[clap(author, version, about, long_about= None)]
struct Args {
    #[clap(
        short = 'r',
        long,
        help = "Parse catalog config instead of using ~/.icebergrc"
    )]
    rc: Option<String>,

    #[clap(long, value_enum, default_value_t = PrintFormat::Automatic)]
    format: PrintFormat,

    #[clap(
        short,
        long,
        help = "Reduce printing other than the results and work quietly"
    )]
    quiet: bool,

    #[clap(
        long,
        help = "The max number of rows to display for 'Table' format\n[possible values: numbers(0/10/...), inf(no limit)]",
        default_value = "40"
    )]
    maxrows: MaxRows,

    #[clap(long, help = "Enables console syntax highlighting")]
    color: bool,
}

#[tokio::main]
/// Calls [`main_inner`], then handles printing errors and returning the correct exit code
pub async fn main() -> ExitCode {
    tracing_subscriber::fmt::init();

    if let Err(e) = main_inner().await {
        println!("Error: {e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

async fn main_inner() -> anyhow::Result<()> {
    let args = Args::parse();

    if !args.quiet {
        println!("ICEBERG PLAYGROUND v{ICEBERG_PLAYGROUND_VERSION}");
    }

    let session_config = SessionConfig::from_env()?.with_information_schema(true);

    let rt_builder = RuntimeEnvBuilder::new();

    let runtime_env = rt_builder.build_arc()?;

    // enable dynamic file query
    let ctx = SessionContext::new_with_config_rt(session_config, runtime_env).enable_url_table();
    ctx.refresh_catalogs().await?;

    let rc = match args.rc {
        Some(ref file) => PathBuf::from_str(file)?,
        None => dirs::home_dir()
            .map(|h| h.join(".icebergrc"))
            .ok_or_else(|| anyhow::anyhow!("cannot find home directory"))?,
    };

    let catalogs = Arc::new(IcebergCatalogList::parse(&rc).await?);
    ctx.register_catalog_list(catalogs);

    exec_from_repl(&ctx, &args).await
}

async fn exec_from_repl(ctx: &SessionContext, args: &Args) -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut statement = String::new();

    loop {
        if !args.quiet {
            let prompt = if statement.is_empty() {
                "iceberg> "
            } else {
                "......> "
            };
            write!(stdout, "{prompt}")?;
            stdout.flush()?;
        }

        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            if statement.trim().is_empty() {
                break;
            }
            anyhow::bail!("incomplete SQL statement at end of input");
        }

        let trimmed = line.trim();
        if statement.is_empty() && matches!(trimmed, "exit" | "quit" | "\\q") {
            break;
        }

        statement.push_str(&line);
        if !trimmed.ends_with(';') {
            continue;
        }

        let sql = statement.trim();
        let sql = sql.strip_suffix(';').unwrap_or(sql).trim();
        if !sql.is_empty() {
            execute_sql(ctx, sql, args).await?;
        }
        statement.clear();
    }

    Ok(())
}

async fn execute_sql(ctx: &SessionContext, sql: &str, args: &Args) -> anyhow::Result<()> {
    let dataframe = ctx.sql(sql).await?;
    match args.format {
        PrintFormat::Automatic | PrintFormat::Table => match args.maxrows {
            MaxRows::Limited(maxrows) => dataframe.show_limit(maxrows).await?,
            MaxRows::Unlimited => dataframe.show().await?,
        },
        PrintFormat::Csv => print_delimited(dataframe, b',', args.maxrows).await?,
        PrintFormat::Tsv => print_delimited(dataframe, b'\t', args.maxrows).await?,
        PrintFormat::Json => print_json(dataframe, args.maxrows, false).await?,
        PrintFormat::Ndjson => print_json(dataframe, args.maxrows, true).await?,
    }

    Ok(())
}

async fn collect_batches(
    dataframe: DataFrame,
    maxrows: MaxRows,
) -> datafusion::error::Result<Vec<RecordBatch>> {
    match maxrows {
        MaxRows::Limited(maxrows) => dataframe.limit(0, Some(maxrows))?.collect().await,
        MaxRows::Unlimited => dataframe.collect().await,
    }
}

async fn print_delimited(
    dataframe: DataFrame,
    delimiter: u8,
    maxrows: MaxRows,
) -> anyhow::Result<()> {
    let batches = collect_batches(dataframe, maxrows).await?;
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let mut writer = CsvWriterBuilder::new()
        .with_header(true)
        .with_delimiter(delimiter)
        .build(&mut handle);

    for batch in &batches {
        writer.write(batch)?;
    }

    Ok(())
}

async fn print_json(
    dataframe: DataFrame,
    maxrows: MaxRows,
    line_delimited: bool,
) -> anyhow::Result<()> {
    let batches = collect_batches(dataframe, maxrows).await?;
    let stdout = io::stdout();
    let mut handle = stdout.lock();

    if line_delimited {
        let mut writer = LineDelimitedWriter::new(&mut handle);
        for batch in &batches {
            writer.write(batch)?;
        }
        writer.finish()?;
    } else {
        let mut writer = ArrayWriter::new(&mut handle);
        for batch in &batches {
            writer.write(batch)?;
        }
        writer.finish()?;
        writeln!(handle)?;
    }

    Ok(())
}
