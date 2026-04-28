use anyhow::Result;
use env_logger;
use log::{debug, info, warn};
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use regex::Regex;
use rten::Model;
use std::path::Path;
use std::path::PathBuf;

#[allow(unused)]
use rten_tensor::prelude::*;

fn file_path(path: &str) -> PathBuf {
    let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    abs_path.push(path);
    abs_path
}

fn recognize_plate(img_path: &str) -> Result<Option<String>> {
    info!("Processando imagem: {}", img_path);
    // Pré-processa a imagem

    let detection_model_path = file_path("models/text-detection.rten");
    let rec_model_path = file_path("models/text-recognition.rten");

    let detection_model = Model::load_file(detection_model_path)?;
    let recognition_model = Model::load_file(rec_model_path)?;

    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })?;

    let img = image::open(img_path).map(|image| image.into_rgb8())?;
    let img_source = ImageSource::from_bytes(img.as_raw(), img.dimensions())?;
    let ocr_input = engine.prepare_input(img_source)?;

    let word_rects = engine.detect_words(&ocr_input)?;

    let line_rects = engine.find_text_lines(&ocr_input, &word_rects);

    let line_texts = engine.recognize_text(&ocr_input, &line_rects)?;
    let full_text: String = line_texts
        .iter()
        .flatten()
        .filter(|l| l.to_string().len() > 1)
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join("");

    info!("Texto bruto do OCR: '{}'", full_text);

    let cleaned = clean_plate_text(&full_text);
    debug!("Texto limpo: '{}'", cleaned);

    if let Some(plate) = validate_plate(&cleaned) {
        let formatted = format_plate_display(&plate);
        info!("✅ Placa reconhecida: {}", formatted);
        Ok(Some(formatted))
    } else {
        warn!("Formato inválido para placa: '{}'", cleaned);
        Ok(None)
    }
}

/// Limpa o texto extraído para formato de placa
fn clean_plate_text(text: &str) -> String {
    // Pega apenas caracteres alfanuméricos e converte para maiúsculas
    let cleaned: String = text
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    cleaned
}

fn validate_plate(plate: &str) -> Option<String> {
    let mercosul_re = Regex::new(r"([A-Z]{3}\d[A-Z]\d{2})").unwrap();
    let re = Regex::new(r"([A-Z0-9]{3})(\d{4})").unwrap();

    if let Some(captures) = mercosul_re.captures(&plate) {
        let plate_raw = &captures[1];
        let prefix = &plate_raw[0..3];
        let suffix = &plate_raw[3..];
        return Some(format!("{}-{}", prefix, suffix));
    }

    if let Some(captures) = re.captures(&plate) {
        let prefix = &captures[1];
        let suffix = &captures[2];

        // Corrige 0 para O se estiver no prefixo (primeiros 3 caracteres)
        let prefix_corrected = if prefix.starts_with('0') && prefix.len() == 3 {
            prefix.replacen('0', "O", 1)
        } else {
            prefix.to_string()
        };

        return Some(format!("{}-{}", prefix_corrected, suffix));
    }

    None
}

/// Formata a placa para exibição (adiciona hífen se necessário)
fn format_plate_display(plate: &str) -> String {
    let plate_clean = plate.replace('-', "");
    if plate_clean.len() == 7 {
        format!("{}-{}", &plate_clean[0..3], &plate_clean[3..])
    } else {
        plate.to_string()
    }
}

fn main() -> anyhow::Result<()> {
    // Inicializa logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("🔍 LEITOR DE PLACAS DE VEÍCULOS - RUST\n");
    println!("=====================================\n");

    // Pega argumentos da linha de comando
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Uso:");
        println!(
            "  {} <arquivo.png/jpg>        - Processa uma imagem",
            args[0]
        );
        println!(
            "  {} --batch <pasta>          - Processa todas imagens de uma pasta",
            args[0]
        );
        println!("  {} --help                   - Mostra esta ajuda", args[0]);
        return Ok(());
    }

    match args[1].as_str() {
        "--batch" => {
            println!("Ainda nao implementado");
        }
        "--help" | "-h" => {
            println!("Reconhece placas de veículos em imagens usando OCR");
            println!("Leitor de Placas de Veículos em Rust");
            println!("\nExemplos:");
            println!("  {} placa.jpg", args[0]);
            println!("  {} --batch ./fotos", args[0]);
        }
        _ => {
            let img_path = &args[1];
            print!("{}", img_path);
            if !Path::new(img_path).exists() {
                println!("❌ Arquivo não encontrado: {}", img_path);
                return Ok(());
            }

            match recognize_plate(img_path) {
                Ok(Some(plate)) => {
                    println!("\n✅ SUCESSO!");
                    println!("🔢 Placa: {}", plate);
                }
                Ok(None) => {
                    println!("\n❌ FALHA!");
                    println!("Não foi possível reconhecer a placa.");
                    println!("\nDicas:");
                    println!("  - Certifique-se que a placa está bem iluminada");
                    println!("  - A imagem deve ter a placa em bom foco");
                    println!("  - Tente cortar a imagem para mostrar só a placa");
                    println!("  - Placas muito inclinadas podem atrapalhar");
                }
                Err(e) => {
                    println!("\n❌ ERRO: {}", e);
                }
            }
        }
    }
    Ok(())
}
