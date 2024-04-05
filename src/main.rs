use std::{collections::HashMap, path::{Path, PathBuf}, sync::Arc};

use tokio::{io::AsyncWriteExt, sync::Semaphore};
use uuid::Uuid;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    use futures::StreamExt;

    let f = tokio::fs::read("horizons.json").await?;
    let files: HashMap<String, serde::de::IgnoredAny> = serde_json::from_slice(&f)?;

    let sema = Arc::new(Semaphore::new(32));

    let client = small_gcs::SharedClient::new("mysticetus-frame-staging").await?;

    let futs = futures::stream::FuturesUnordered::new();

    for file in files.into_keys() {
        
        let client = client.clone();
        let sema = Arc::clone(&sema);

        futs.push(tokio::spawn(async move {
            let permit = sema.acquire_owned().await.unwrap();

            let dst = PathBuf::from(format!("images/{file}"));

            if let Some(parent) = dst.parent() {
                if !tokio::fs::try_exists(parent).await? {
                    tokio::fs::create_dir_all(&parent).await?;
                }
            }
            
            let mut dst = tokio::fs::File::create(&dst).await?;

            println!("copying {file}...");

            let stream = client.read(&file).content().await?;
            futures::pin_mut!(stream);

            while let Some(chunk) = stream.next().await {
                let mut chunk = chunk?;

                dst.write_all_buf(&mut chunk).await?;
            }


            dst.sync_all().await?;

            drop(permit);
            Ok(()) as anyhow::Result<()>

        }));
    }

    futures::pin_mut!(futs);

    while let Some(res) = futs.next().await {
        res??;
    }

    Ok(())
}



#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameDoc {
    frame_path: String,
    offset: f64,
    #[serde(default)]
    annotations: HashMap<Uuid, Annotation>,
    #[serde(default)]
    annotated_by: Vec<String>,
    video_slug: String,
    frame_slug: String,
}

impl FrameDoc {
    fn read_from<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;

        serde_json::from_slice(&bytes).map_err(anyhow::Error::from)
    }
}


#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Annotation {
    id: Uuid,
    #[serde(flatten)]
    kind: AnnotationKind,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
enum AnnotationKind {
    Horizon(Horizon),
    AnimalDetection(AnimalDetection),
    Vessel(Vessel),
}


#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Horizon {
    start: Point,
    end: Point,
    #[serde(default)]
    control_points: Vec<Point>,
}

impl Horizon {
    fn into_better(self) -> BetterHorizon {
        let control = match self.control_points.len() {
            0 => Point {
                x: (self.start.x + self.end.x) / 2.0,
                y: (self.start.y + self.end.y) / 2.0,
            },
            1 => self.control_points[0],
            uhh => panic!("found {uhh} ctrl points?"),
        };

        BetterHorizon { start: self.start, control, end: self.end }
    }
}


#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BetterHorizon {
    start: Point,
    control: Point,
    end: Point,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimalDetection {
    bounding_box: BoundingBox,
    detection_type: DetectionType,
}


#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Vessel {
    bounding_box: BoundingBox,
}
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DetectionType {
    Body,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
struct Size {
    height: f64,
    width: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
struct BoundingBox {
    size: Size,
    point: Point,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() -> anyhow::Result<()> {
        let f = FrameDoc::read_from("/home/mrudisel/src/horizon/data/0A07Ff2ync.json")?;
        println!("{f:#?}");
        Ok(())   
    }

    #[test]
    fn test_all() -> anyhow::Result<()> {
        let mut c = 0;

        let mut horizons = HashMap::with_capacity(8000);

        for res in std::fs::read_dir("/home/mrudisel/src/horizon/data/")? {
            let entry = res?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("json")) {
                let doc = FrameDoc::read_from(&path)
                    .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))?;
                

                let horiz = doc.annotations.into_values().find_map(|anno| match anno.kind {
                    AnnotationKind::Horizon(horiz) => Some(horiz),
                    _ => None,
                });

                if let Some(horiz) = horiz {
                    horizons.insert(doc.frame_path, horiz.into_better());
                }
                
                c += 1;
            }
        }

        println!("found {} horizons", horizons.len());

        let mut f = std::fs::File::create("horizons.json")?;
        serde_json::to_writer_pretty(&mut f, &horizons)?;
        
        println!("successfully parsed {c} files");
        Ok(())
    }
}


