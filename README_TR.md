# Ollama Araç Fonksiyonları Uygulama Kılavuzu

Bu kılavuz, Ollama'nın LLM modelleriyle araç fonksiyonlarının nasıl uygulanacağını ve kullanılacağını açıklar.

## Ön Koşullar

1. Ollama kurulu ve çalışır durumda
2. Python 3.7+
3. Gerekli Python paketleri:
```bash
pip install ollama requests pandas yfinance nltk
```

## Yapılandırma

Örnekleri kullanmadan önce:

1. `config.py` dosyasını ayarlarınızla oluşturun:
```python
from config import OLLAMA_CONFIG, API_KEYS, MODEL_PARAMS, TOOL_FUNCTIONS, API_ENDPOINTS, ERROR_MESSAGES
```

2. Yapılandırmayı gerçek değerlerinizle güncelleyin:
   - Ollama sunucu URL'nizi `OLLAMA_CONFIG["base_url"]` içinde ayarlayın
   - API anahtarlarınızı `API_KEYS` içine ekleyin
   - Gerekirse `MODEL_PARAMS` içindeki model parametrelerini ayarlayın

## Temel Örnek

```python
import ollama
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# Ollama örneğiniz için temel URL'yi yapılandırın
ollama.set_host(OLLAMA_CONFIG["base_url"])

response = ollama.chat(
    model=OLLAMA_CONFIG["default_model"],
    messages=[{'role': 'user', 'content': 'Toronto\'da hava nasıl?'}],
    tools=[{
      'type': 'function',
      'function': TOOL_FUNCTIONS["get_current_weather"]
    }],
)

print(response['message']['tool_calls'])
```

> [!NOTE]
> Örnekleri çalıştırmadan önce `config.py` dosyasını gerçek Ollama sunucu URL'niz ve API anahtarlarınızla güncellediğinizden emin olun.

## İçindekiler

1. [Giriş](#giriş)
2. [Ollama'nın Araç Fonksiyonları Mimarisi](#ollamanın-araç-fonksiyonları-mimarisi)
3. [Ollama'daki Şablon Sistemi](#ollamadaki-şablon-sistemi)
4. [Araç Fonksiyonları Uygulama Detayları](#araç-fonksiyonları-uygulama-detayları)
5. [LLM'ler Araç Çağrılarını Nasıl Üretir](#llmler-araç-çağrılarını-nasıl-üretir)
6. [LLM Çıktısından Araç Çağrılarını Ayrıştırma](#llm-çıktısından-araç-çağrılarını-ayrıştırma)
7. [API Entegrasyonu](#api-entegrasyonu)
8. [Pratik Örnekler](#pratik-örnekler)
9. [Araç Yürütme İş Akışı](#araç-yürütme-iş-akışı)
10. [Çoklu Araç Kullanımı](#çoklu-araç-kullanımı)
11. [Hiyerarşik Ajan Araç Kullanımı](#hiyerarşik-ajan-araç-kullanımı)
12. [Gerçek Çalıştırılabilir Örnekler](#gerçek-çalıştırılabilir-örnekler)
13. [Sonuç](#sonuç)

## Giriş

Ollama, büyük dil modellerini (LLM) yerel olarak çalıştırmaya olanak sağlayan açık kaynaklı bir çerçevedir. Gelişmiş özelliklerinden biri olan araç fonksiyonları desteği, LLM'lerin metin üretimi ötesinde görevler gerçekleştirmek için harici araçlar ve API'lerle etkileşime girmesini sağlar.

Bu belge, Ollama'nın araç fonksiyonlarını nasıl uyguladığını, şablon sisteminden model çıktılarından araç çağrılarını çıkaran ayrıştırma mekanizmasına kadar incelemektedir.

## Ollama'nın Araç Fonksiyonları Mimarisi

Ollama'nın araç fonksiyonu yeteneği birkaç temel bileşen üzerine inşa edilmiştir:

1. **API Tipleri**: Araçlarla ilgili yapıların tanımları (`api/types.go` içinde)
2. **Şablon Sistemi**: Araç çağrı desteği içeren özelleştirilebilir şablonlar (`template` paketi içinde)
3. **Ayrıştırma Mantığı**: Model çıktılarından araç çağrılarını çıkaran kod (`server/model.go` içinde)
4. **API Entegrasyonu**: Araçlar için OpenAI uyumlu uç noktalar (`openai/openai.go` içinde)

Mimari, modüler bir tasarım izler:
- Model şablonları araç istemlerinin nasıl biçimlendirileceğini belirler
- Sunucu LLM yanıtlarını işleyerek araç çağrılarını tanımlar ve çıkarır
- API katmanı istemciler için standart bir arayüz sağlar

## Ollama'daki Şablon Sistemi

Ollama, farklı LLM'ler için istemleri biçimlendirmek üzere Go'nun şablon sistemini kullanır. Şablonlar, araç fonksiyonu desteği için önemlidir çünkü araç tanımlarının modele nasıl sunulacağını belirler.

Ollama'daki şablonlar:
- Go'nun `text/template` paketi kullanılarak tanımlanır
- İsteğe bağlı bileşenleri işlemek için `{{ if .System }}` gibi koşulları içerebilir
- Araç entegrasyonu için `.ToolCalls` gibi özel değişkenleri destekler

Araçları destekleyen örnek bir şablon şöyle görünebilir:

```go
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}{{ if .ToolCalls }}{{ range .ToolCalls }}
<|im_start|>tool
{{ .Function.Name }}({{ .Function.Arguments }})
<|im_end|>
{{ end }}{{ end }}<|im_start|>assistant
{{ .Response }}"""
```

Şablonlar modelin meta verilerinde saklanır ve çıkarım sırasında istemleri düzgün bir şekilde yapılandırmak için kullanılır.

## Araç Fonksiyonları Uygulama Detayları

Ollama'daki araç fonksiyonları birkaç temel yapı aracılığıyla temsil edilir:

```go
type Tool struct {
    Type     string       `json:"type"`
    Function ToolFunction `json:"function"`
}

type ToolFunction struct {
    Name        string `json:"name"`
    Description string `json:"description"`
    Parameters  struct {
        Type       string   `json:"type"`
        Required   []string `json:"required"`
        Properties map[string]struct {
            Type        string   `json:"type"`
            Description string   `json:"description"`
            Enum        []string `json:"enum,omitempty"`
        } `json:"properties"`
    } `json:"parameters"`
}

type ToolCall struct {
    Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
    Name      string                    `json:"name"`
    Arguments ToolCallFunctionArguments `json:"arguments"`
}

type ToolCallFunctionArguments map[string]any
```

Bu yapılar şunları sağlar:
1. Modelin kullanabileceği mevcut araçları tanımlama
2. Model tarafından yapılan araç çağrılarını temsil etme
3. Bu araçlara argümanları iletme

## LLM'ler Araç Çağrılarını Nasıl Üretir

Bir LLM araç tanımlarıyla istemlendiğinde, hangi aracı çağıracağını ve hangi parametrelerle çağıracağını belirten yapılandırılmış bir çıktı üretmesi gerekir. Bu şu şekilde gerçekleşir:

1. **İstem Mühendisliği**: Şablon, modele mevcut araçlar hakkında talimat vermek için istemi biçimlendirir
2. **JSON Üretimi**: Model, Ollama'nın ayrıştırabileceği araç çağrıları içeren JSON formatında bir yanıt üretir
3. **Çıktı Biçimlendirme**: Model, yanıtını yapılandırmak için belirtilen şablonu takip eder

Llama 3.1, Mistral Nemo ve Firefunction v2 gibi modeller, Ollama'nın ayrıştırabileceği bir formatta araç çağrıları üretmek için özel olarak eğitilmiştir.

## LLM Çıktısından Araç Çağrılarını Ayrıştırma

Ollama'nın araç fonksiyonu uygulamasının kalbi, `server/model.go` içindeki `parseToolCalls` metodudur. Bu fonksiyon, LLM'nin çıktı metninden yapılandırılmış araç çağrılarını çıkarır:

```go
// parseToolCalls, bir JSON dizesini ToolCalls dilimine ayrıştırmaya çalışır.
// mxyng: bu sadece girdi JSON formatında araç çağrıları içeriyorsa çalışır
func (m *Model) parseToolCalls(s string) ([]api.ToolCall, bool) {
    // Adım 1: Araç çağrılarını işleyen şablon alt ağacını bul
    tmpl := m.Template.Subtree(func(n parse.Node) bool {
        if t, ok := n.(*parse.RangeNode); ok {
            return slices.Contains(template.Identifiers(t.Pipe), "ToolCalls")
        }
        return false
    })
    if tmpl == nil {
        return nil, false
    }

    // Adım 2: Yapısını anlamak için şablonu sahte verilerle yürüt
    var b bytes.Buffer
    if err := tmpl.Execute(&b, map[string][]api.ToolCall{
        "ToolCalls": {
            {
                Function: api.ToolCallFunction{
                    Name: "@@name@@",
                    Arguments: api.ToolCallFunctionArguments{
                        "@@argument@@": 1,
                    },
                },
            },
        },
    }); err != nil {
        return nil, false
    }

    // Adım 3: Şablon nesnelerini analiz et ve alan eşlemelerini bul
    templateObjects := parseObjects(b.String())
    if len(templateObjects) == 0 {
        return nil, false
    }
    
    // Adım 4: Hangi alanların ad ve argümanlara karşılık geldiğini belirle
    var name, arguments string
    for k, v := range templateObjects[0] {
        switch v.(type) {
        case string:
            name = k
        case map[string]any:
            arguments = k
        }
    }
    if name == "" || arguments == "" {
        return nil, false
    }

    // Adım 5: Gerçek model yanıtını ayrıştır
    responseObjects := parseObjects(s)
    if len(responseObjects) == 0 {
        return nil, false
    }

    // Adım 6: Tüm iç içe nesneleri özyinelemeli olarak topla
    var collect func(any) []map[string]any
    collect = func(obj any) (all []map[string]any) {
        switch o := obj.(type) {
        case map[string]any:
            all = append(all, o)
            for _, v := range o {
                all = append(all, collect(v)...)
            }
        case []any:
            for _, v := range o {
                all = append(all, collect(v)...)
            }
        }
        return all
    }
    
    var objs []map[string]any
    for _, p := range responseObjects {
        objs = append(objs, collect(p)...)
    }

    // Adım 7: Eşleşen nesnelerden araç çağrılarını çıkar
    var toolCalls []api.ToolCall
    for _, kv := range objs {
        n, nok := kv[name].(string)
        a, aok := kv[arguments].(map[string]any)
        if nok && aok {
            toolCalls = append(toolCalls, api.ToolCall{
                Function: api.ToolCallFunction{
                    Name: n,
                    Arguments: a,
                },
            })
        }
    }

    return toolCalls, len(toolCalls) > 0
}
```

Bu karmaşık ayrıştırma süreci:

1. **Şablon analizi kullanır**: Modelin şablonuna göre araç çağrılarını nasıl yapılandırdığını belirler
2. **Desen eşleştirme uygular**: Fonksiyon adlarını ve argüman haritalarını tanımlar
3. **İç içe yapıları işler**: Karmaşık JSON çıktılarını özyinelemeli olarak gezinir
4. **Yapılandırılmış nesneler oluşturur**: Model çıktısını kullanılabilir `ToolCall` yapılarına dönüştürür

Ayrıştırıcı, araç çağrılarının temel bileşenlerini (fonksiyon adı ve argümanlar) içerdiği sürece farklı model-spesifik formatları işleyebilecek kadar esnektir.

## API Entegrasyonu

Ollama, araç fonksiyonu yeteneğini OpenAI uyumlu uç noktalar dahil olmak üzere API'si üzerinden sunar. Temel bileşenler şunları içerir:

1. **İstek İşleme**: Araç tanımlarıyla gelen istekleri işleme
2. **Yanıt Biçimlendirme**: Araç çağrılarını içeren yanıtları yapılandırma
3. **Ara Yazılım**: Ollama'nın iç formatı ile OpenAI uyumlu format arasında dönüşüm

OpenAI uyumluluk katmanı özellikle önemlidir, mevcut araçlar ve kütüphanelerle sorunsuz entegrasyona olanak sağlar.

## Pratik Örnekler

### Python Örneği

```python
import ollama
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# Ollama örneğiniz için temel URL'yi yapılandırın
ollama.set_host(OLLAMA_CONFIG["base_url"])

response = ollama.chat(
    model=OLLAMA_CONFIG["default_model"],
    messages=[{'role': 'user', 'content': 'Toronto\'da hava nasıl?'}],
    tools=[{
      'type': 'function',
      'function': TOOL_FUNCTIONS["get_current_weather"]
    }],
)

print(response['message']['tool_calls'])
```

### OpenAI Uyumluluk Örneği

```python
import openai
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# Ollama örneğiniz için temel URL'yi yapılandırın
openai.base_url = f"{OLLAMA_CONFIG['base_url']}/v1"
openai.api_key = 'ollama'

response = openai.chat.completions.create(
    model=OLLAMA_CONFIG["default_model"],
    messages=[{"role": "user", "content": "New York'ta hava nasıl?"}],
    tools=[{
      "type": "function",
      "function": TOOL_FUNCTIONS["get_current_weather"]
    }],
)

print(response.choices[0].message.tool_calls)
```

> [!NOTE]
> Örnekleri çalıştırmadan önce `config.py` dosyasını gerçek Ollama sunucu URL'niz ve API anahtarlarınızla güncellediğinizden emin olun.

## Araç Yürütme İş Akışı

Ollama'nın araç fonksiyonu uygulamasıyla ilgili kritik bir nokta, **Ollama'nın araçları doğrudan yürütmemesidir**. Ollama sadece şunları yapar:

1. Araç tanımlarını istemlere biçimlendirme
2. Model çıktılarından araç çağrılarını ayrıştırma
3. Bu yapılandırılmış araç çağrılarını istemci uygulamaya sağlama

Araçların gerçek yürütülmesi istemci uygulamanın sorumluluğundadır. İşte araç yürütme için tam iş akışı:

```
┌─────────────┐    1. Araç tanımları    ┌────────┐    2. Araç istemi      ┌─────┐
│ Uygulamanız │─────────────────────────▶ Ollama │───────────────────────▶ LLM │
└─────┬───────┘                         └────┬───┘                        └──┬──┘
      │                                      │                               │
      │                                      │     3. Araç çağrısı içeren    │
      │                                      │        ham metin              │
      │                                      │◀─────────────────────────────┘
      │      4. Ayrıştırılmış araç çağrısı  │
      │◀───────────────────────────────────────┘
      │
      │      5. Uygulamanızda aracı yürüt
      ├─────────────────────┐
      │                     │
      │                     ▼
      │      6. Araç sonucunu al
      │◀────────────────────┘
      │
      │      7. Sonucu LLM'e geri gönder
      ▼
┌─────────────┐    Araç sonuç mesajı    ┌────────┐       Sonucu biçimlendir      ┌─────┐
│ Uygulamanız │─────────────────────────▶ Ollama │──────────────────────────▶ LLM │
└─────────────┘                         └────────┘                           └─────┘
```

### Uygulama Adımları

1. **Araçları Tanımla**: Uygulamanız araçları ve parametrelerini tanımlar
2. **İstek Gönder**: Araçlar isteminizle birlikte Ollama'ya gönderilir
3. **Yanıtı Ayrıştır**: Ollama modelin yapılandırılmış araç çağrısını döndürür
4. **Aracı Yürüt**: Uygulamanız belirtilen fonksiyonu sağlanan argümanlarla yürütür
5. **Sonuçları Döndür**: Aracın çıktısını "tool" rolüyle bir mesaj olarak Ollama'ya geri gönder

### Örnek Yürütme Akışı

```python
import ollama
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# 1. Uygulamanızda araç uygulamanızı tanımlayın
def get_weather(city):
    # Gerçek API çağrısı veya fonksiyon uygulaması
    return f"{city}'de hava 22°C ve güneşli"

# 2. LLM için aracı tanımlayın
response = ollama.chat(
    model=OLLAMA_CONFIG["default_model"],
    messages=[{'role': 'user', 'content': 'Toronto\'da hava nasıl?'}],
    tools=[{
      'type': 'function',
      'function': TOOL_FUNCTIONS["get_current_weather"]
    }],
)

# 3. LLM'in bir araç kullanmak isteyip istemediğini kontrol et
if 'tool_calls' in response['message']:
    tool_calls = response['message']['tool_calls']
    for tool_call in tool_calls:
        # 4. Aracı uygulamanızda çalıştırın
        if tool_call['function']['name'] == 'get_current_weather':
            city = tool_call['function']['arguments']['city']
            weather_info = get_weather(city)  # Bu sizin kodunuzda gerçekleşir!
            
            # 5. Sonucu LLM'e geri gönderin
            final_response = ollama.chat(
                model=OLLAMA_CONFIG["default_model"],
                messages=[
                    {'role': 'user', 'content': 'Toronto\'da hava nasıl?'},
                    response['message'],
                    {'role': 'tool', 'content': weather_info}  # Araç sonucu
                ]
            )
            print(final_response['message']['content'])
```

## Çoklu Araç Kullanımı

Ollama, bir modelin tek bir yanıtta veya bir konuşma boyunca birden fazla araç çağırabileceği senaryoları destekler. İşte çoklu araç kullanımını uygulama yöntemi:

### Tek Çağrıda Çoklu Araç

Bazı modeller tek bir yanıtta birden fazla araç çağrısı döndürebilir. Uygulamanızın her araç çağrısını ayrı ayrı işlemesi gerekir:

```python
import ollama
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# Araçların uygulaması
def get_weather(city):
    return f"{city}'de hava 22°C ve güneşli"

def get_population(city):
    populations = {
        "Toronto": "2.93 milyon",
        "New York": "8.8 milyon",
        "London": "8.9 milyon"
    }
    return populations.get(city, f"{city} için nüfus verisi mevcut değil")

# Birden fazla aracı tanımlayın
response = ollama.chat(
    model=OLLAMA_CONFIG["default_model"],
    messages=[{'role': 'user', 'content': 'Toronto\'da hava ve nüfus nasıl?'}],
    tools=[
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["get_current_weather"]
        },
        {
            'type': 'function',
            'function': TOOL_FUNCTIONS["get_population"]
        }
    ],
)

# Tüm araç çağrılarını işle
if 'tool_calls' in response['message']:
    tool_calls = response['message']['tool_calls']
    
    # Geri gönderilecek tüm mesajları sakla
    conversation = [
        {'role': 'user', 'content': 'Toronto\'da hava ve nüfus nasıl?'},
        response['message']
    ]
    
    # Her araç çağrısını işle
    for tool_call in tool_calls:
        function_name = tool_call['function']['name']
        args = tool_call['function']['arguments']
        
        if function_name == 'get_current_weather':
            result = get_weather(args['city'])
            conversation.append({'role': 'tool', 'content': result})
        
        elif function_name == 'get_population':
            result = get_population(args['city'])
            conversation.append({'role': 'tool', 'content': result})
    
    # Tüm araç sonuçlarını modele gönder
    final_response = ollama.chat(
        model=OLLAMA_CONFIG["default_model"],
        messages=conversation
    )
    
    print(final_response['message']['content'])
```

### Sıralı Araç Çağrıları

Daha karmaşık senaryolarda, model önceki sonuçlara dayanarak bir dizi araç çağrısı yapabilir:

```python
import ollama
from config import OLLAMA_CONFIG, TOOL_FUNCTIONS

# Araç uygulamaları
def search_database(query):
    # Simüle edilmiş veritabanı araması
    if "product" in query:
        return "Bulunan ürünler: Widget A, Widget B ve Widget C"
    return "Arama sonucu bulunamadı: " + query

def get_product_details(product_id):
    # Simüle edilmiş ürün detayları araması
    products = {
        "Widget A": {"price": "$10.99", "stock": 42, "category": "Araçlar"},
        "Widget B": {"price": "$24.99", "stock": 7, "category": "Elektronik"},
        "Widget C": {"price": "$5.50", "stock": 0, "category": "Ofis Malzemeleri"}
    }
    return str(products.get(product_id, "Ürün bulunamadı"))

# İlk araçlar tanımı
tools = [
    {
        'type': 'function',
        'function': TOOL_FUNCTIONS["search_database"]
    },
    {
        'type': 'function',
        'function': TOOL_FUNCTIONS["get_product_details"]
    }
]

# İlk konuşma
conversation = [
    {'role': 'user', 'content': 'Stokta olan widget\'lar hakkında detay istiyorum'}
]

# İlk API çağrısı - model muhtemelen önce arama yapacak
response = ollama.chat(
    model=OLLAMA_CONFIG["default_model"],
    messages=conversation,
    tools=tools
)

# Araç çağrılarını işle
if 'tool_calls' in response['message']:
    conversation.append(response['message'])
    
    for tool_call in response['message']['tool_calls']:
        function_name = tool_call['function']['name']
        args = tool_call['function']['arguments']
        
        if function_name == 'search_database':
            search_results = search_database(args['query'])
            conversation.append({'role': 'tool', 'content': search_results})
        
        elif function_name == 'get_product_details':
            product_details = get_product_details(args['product_id'])
            conversation.append({'role': 'tool', 'content': product_details})

    # Son yanıtı al
    final_response = ollama.chat(
        model=OLLAMA_CONFIG["default_model"],
        messages=conversation
    )
    print(final_response['message']['content'])

# Örnek kullanım
if __name__ == "__main__":
    # API anahtarlarını config'den al
    OPENWEATHER_API_KEY = API_KEYS["openweather"]
    ALPHA_VANTAGE_API_KEY = API_KEYS["alpha_vantage"]
    NEWS_API_KEY = API_KEYS["news_api"]
    
    # Örnek sorgular
    queries = [
        "Tokyo'da hava nasıl ve kuantum bilgisayarlar konusunda son gelişmeler neler?",
        "İklim değişikliğinin kutup ayıları üzerindeki etkisini araştır ve bulguları analiz et",
        "AB'de yapay zeka düzenlemelerinin mevcut durumu nedir ve başlangıç şirketlerini nasıl etkiliyor?"
    ]
    
    for query in queries:
        print(f"\nSorgu işleniyor: {query}")
        execute_research_task(query)
```

### Çoklu Araç Kullanımı için Önemli Noktalar

1. **Araç Önceliği**: LLM, isteğin doğasına göre hangi aracı çağıracağına karar verir
2. **Araç Bağımlılıkları**: Bazı görevler, sonraki çağrıların önceki sonuçlara bağlı olduğu sıralı araç çağrıları gerektirir
3. **Konuşma Yönetimi**: Tüm konuşmayı, tüm araç çağrıları ve yanıtları dahil olmak üzere takip edin
4. **Hata Yönetimi**: Araç yürütmesinin başarısız olduğu durumlar için sağlam hata yönetimi uygulayın
5. **Paralel vs. Sıralı**: Birden fazla araç çağrısını paralel mi yoksa sırayla mı yürüteceğinize karar verin

Çoklu araç kullanımı, LLM'lerin karmaşık görevleri daha basit işlemlere bölerek, her birinin özel araçlar tarafından işlenerek gerçekleştirmesine olanak sağlar.

## Hiyerarşik Ajan Araç Kullanımı

Ollama'nın araç fonksiyonu sistemi, bir ajanın görevleri diğer uzman ajanlara devredebileceği hiyerarşik ajanları destekleyecek şekilde genişletilebilir. Bu, karmaşık görev ayrıştırma ve paralel işleme sağlar. İşte hiyerarşik ajan araç kullanımını uygulama yöntemi:

### Ajan Hiyerarşi Yapısı

```
┌─────────────────┐
│  Ana Ajan       │
│  (Koordinatör)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Alt-Ajan 1     │
│  (Uzman)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Alt-Ajan 2     │
│  (Uzman)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Araç Yürütme   │
│  (Gerçek İş)    │
└─────────────────┘
```

### Uygulama Örneği

```python
import ollama
from typing import List, Dict, Any

class Agent:
    def __init__(self, name: str, role: str, tools: List[Dict[str, Any]]):
        self.name = name
        self.role = role
        self.tools = tools
        self.conversation = []

    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})

    def chat(self, model: str = 'llama3.1') -> Dict[str, Any]:
        response = ollama.chat(
            model=model,
            messages=self.conversation,
            tools=self.tools
        )
        self.add_message("assistant", response['message']['content'])
        return response

class CoordinatorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Koordinatör",
            role="Görev koordinatörü ve devredici",
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'delegate_task',
                        'description': 'Bir görevi uzman bir ajana devret',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'task': {
                                    'type': 'string',
                                    'description': 'Devredilecek görev',
                                },
                                'agent': {
                                    'type': 'string',
                                    'description': 'Görevi işleyecek uzman ajan',
                                },
                                'context': {
                                    'type': 'string',
                                    'description': 'Görev için ek bağlam',
                                }
                            },
                            'required': ['task', 'agent', 'context'],
                        },
                    },
                }
            ]
        )

class ResearchAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Araştırmacı",
            role="Araştırma uzmanı",
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'search_database',
                        'description': 'Veritabanında bilgi ara',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'query': {
                                    'type': 'string',
                                    'description': 'Arama sorgusu',
                                },
                                'filters': {
                                    'type': 'object',
                                    'description': 'Arama filtreleri',
                                }
                            },
                            'required': ['query'],
                        },
                    },
                }
            ]
        )

class AnalysisAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Analist",
            role="Veri analizi uzmanı",
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'analyze_data',
                        'description': 'Sağlanan veriyi analiz et',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'data': {
                                    'type': 'string',
                                    'description': 'Analiz edilecek veri',
                                },
                                'analysis_type': {
                                    'type': 'string',
                                    'description': 'Yapılacak analiz türü',
                                }
                            },
                            'required': ['data', 'analysis_type'],
                        },
                    },
                }
            ]
        )

def execute_hierarchical_task(user_query: str):
    # Ajanları başlat
    coordinator = CoordinatorAgent()
    researcher = ResearchAgent()
    analyst = AnalysisAgent()

    # Koordinatörle başla
    coordinator.add_message("user", user_query)
    coordinator_response = coordinator.chat()

    # Koordinatörün yanıtını işle
    if 'tool_calls' in coordinator_response['message']:
        for tool_call in coordinator_response['message']['tool_calls']:
            if tool_call['function']['name'] == 'delegate_task':
                args = tool_call['function']['arguments']
                task = args['task']
                agent_name = args['agent']
                context = args['context']

                # Görevi uygun ajana yönlendir
                if agent_name == "Araştırmacı":
                    researcher.add_message("user", f"Görev: {task}\nBağlam: {context}")
                    research_response = researcher.chat()
                    
                    if 'tool_calls' in research_response['message']:
                        for research_tool in research_response['message']['tool_calls']:
                            if research_tool['function']['name'] == 'search_database':
                                search_args = research_tool['function']['arguments']
                                # Aramayı yürüt ve sonuçları al
                                search_results = "Örnek arama sonuçları..."  # Gerçek arama ile değiştir
                                
                                # Analizi analiste devret
                                analyst.add_message("user", f"Bu veriyi analiz et: {search_results}")
                                analysis_response = analyst.chat()
                                
                                if 'tool_calls' in analysis_response['message']:
                                    for analysis_tool in analysis_response['message']['tool_calls']:
                                        if analysis_tool['function']['name'] == 'analyze_data':
                                            analysis_args = analysis_tool['function']['arguments']
                                            # Analizi yürüt ve sonuçları al
                                            analysis_results = "Örnek analiz sonuçları..."  # Gerçek analiz ile değiştir
                                            
                                            # Sonuçları koordinatöre gönder
                                            coordinator.add_message("tool", f"Araştırma ve analiz sonuçları: {analysis_results}")
                                            final_response = coordinator.chat()
                                            print(final_response['message']['content'])

# Örnek kullanım
execute_hierarchical_task("Son 5 yıldaki elektrikli araçlar için pazar trendlerini araştır ve analiz et")
```

### Hiyerarşik Ajan Uygulamasının Temel Özellikleri

1. **Ajan Uzmanlaşması**:
   - Her ajanın belirli araçları ve yetenekleri vardır
   - Ajanlar farklı alanlar için tasarlanabilir (araştırma, analiz, karar verme vb.)

2. **Görev Devri**:
   - Koordinatör ajan görevleri uzman ajanlara devredebilir
   - Görevler alt görevlere bölünebilir
   - Sonuçlar toplanabilir ve sentezlenebilir

3. **Konuşma Yönetimi**:
   - Her ajan kendi konuşma geçmişini tutar
   - Sonuçlar yapılandırılmış mesajlar aracılığıyla ajanlar arasında iletilir
   - Bağlam görev yürütme boyunca korunur

4. **Hata Yönetimi ve Kurtarma**:
   - Her ajan hataları bağımsız olarak işleyebilir
   - Başarısız görevler yeniden denenebilir veya alternatif ajanlara devredilebilir
   - Sonuçlar her adımda doğrulanabilir

5. **Ölçeklenebilirlik**:
   - Farklı görev türlerini işlemek için yeni ajanlar eklenebilir
   - Ajanlar farklı hiyerarşilerde organize edilebilir
   - Görevler paralel veya sıralı olarak işlenebilir

### Hiyerarşik Ajan Uygulaması için En İyi Uygulamalar

1. **Net Ajan Rolleri**:
   - Her ajan için belirli sorumluluklar tanımla
   - Açıklayıcı isimler ve roller kullan
   - Ajan yeteneklerini ve sınırlamalarını belgele

2. **Verimli İletişim**:
   - Yapılandırılmış mesaj formatları kullan
   - Devirlerde ilgili bağlamı dahil et
   - Konuşma geçmişini uygun şekilde koru

3. **Görev Ayrıştırma**:
   - Karmaşık görevleri yönetilebilir alt görevlere böl
   - Alt görevler arasındaki bağımlılıkları göz önünde bulundur
   - Sonuç toplama için plan yap

4. **Hata Yönetimi**:
   - Her seviyede sağlam hata yönetimi uygula
   - Yedek mekanizmalar sağla
   - Hataları ve kurtarma girişimlerini kaydet

5. **Performans Optimizasyonu**:
   - Mümkün olduğunda paralel işleme kullan
   - Uygun olduğunda sonuçları önbelleğe al
   - Ajan etkileşimlerini izle ve optimize et

Hiyerarşik ajan araç kullanımı, birden fazla uzman ajanı, her birinin kendi araçları ve yetenekleriyle birlikte kullanarak karmaşık görev işlemeyi sağlar. Bu yaklaşım, farklı uzmanlık türleri veya çoklu işleme adımları gerektiren görevler için özellikle kullanışlıdır.

## Gerçek Çalıştırılabilir Örnekler

Bu bölüm, gerçek API'ler ve servisler kullanan gerçek, çalıştırılabilir örnekler sunar. Bu örnekler verilen kodla doğrudan çalıştırılabilir.

### Örnek 1: Web Arama ve Analiz

Bu örnek, web araması için DuckDuckGo ve hava durumu verisi için OpenWeatherMap kullanır:

```python
import ollama
import requests
from typing import Dict, Any
import json
from datetime import datetime

# API Anahtarları (kendi anahtarlarınızla değiştirin)
OPENWEATHER_API_KEY = "your_openweather_api_key"  # https://openweathermap.org/api adresinden alın

def search_web(query: str) -> str:
    """DuckDuckGo API kullanarak web'de ara"""
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # İlgili bilgileri çıkar
    results = []
    if "Abstract" in data and data["Abstract"]:
        results.append(f"Özet: {data['Abstract']}")
    if "RelatedTopics" in data:
        for topic in data["RelatedTopics"][:3]:
            if "Text" in topic:
                results.append(topic["Text"])
    
    return "\n".join(results) if results else "Sonuç bulunamadı."

def get_weather(city: str) -> str:
    """OpenWeatherMap API kullanarak hava durumu verisi al"""
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code == 200:
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        return f"{city}'de mevcut hava durumu: {description}, Sıcaklık: {temp}°C, Nem: {humidity}%"
    else:
        return f"{city} için hava durumu alınırken hata: {data.get('message', 'Bilinmeyen hata')}"

def analyze_text(text: str) -> str:
    """Ollama kullanarak metni analiz et"""
    response = ollama.chat(
        host="http://your-ollama-url:11434",  # Ollama URL'nizi buraya yazın
        model='llama3.1',
        messages=[{
            'role': 'user',
            'content': f"Bu metni analiz et ve önemli içgörüler sağla:\n\n{text}"
        }]
    )
    return response['message']['content']

def execute_research_task(query: str):
    """Birden fazla araç kullanarak bir araştırma görevi yürüt"""
    # Araçları tanımla
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'search_web',
                'description': 'Web\'de bilgi ara',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'Arama sorgusu',
                        }
                    },
                    'required': ['query'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'description': 'Bir şehir için mevcut hava durumunu al',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'city': {
                            'type': 'string',
                            'description': 'Şehir adı',
                        }
                    },
                    'required': ['city'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'analyze_text',
                'description': 'Metni analiz et ve içgörüler sağla',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'Analiz edilecek metin',
                        }
                    },
                    'required': ['text'],
                },
            },
        }
    ]

    # İlk konuşma
    conversation = [
        {'role': 'user', 'content': query}
    ]

    # İlk API çağrısı
    response = ollama.chat(
        model='llama3.1',
        messages=conversation,
        tools=tools
    )

    # Araç çağrılarını işle
    if 'tool_calls' in response['message']:
        conversation.append(response['message'])
        
        for tool_call in response['message']['tool_calls']:
            function_name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            
            if function_name == 'search_web':
                search_results = search_web(args['query'])
                conversation.append({'role': 'tool', 'content': search_results})
            
            elif function_name == 'get_weather':
                weather_info = get_weather(args['city'])
                conversation.append({'role': 'tool', 'content': weather_info})
            
            elif function_name == 'analyze_text':
                analysis = analyze_text(args['text'])
                conversation.append({'role': 'tool', 'content': analysis})

        # Son yanıtı al
        final_response = ollama.chat(
            model='llama3.1',
            messages=conversation
        )
        print(final_response['message']['content'])

# Örnek kullanım
if __name__ == "__main__":
    # OpenWeatherMap API anahtarınızla değiştirin
    OPENWEATHER_API_KEY = "your_openweather_api_key"
    
    # Örnek sorgular
    queries = [
        "Tokyo'da hava durumu nasıl ve kuantum bilgisayarlarla ilgili son gelişmeler neler?",
        "İklim değişikliğinin kutup ayıları üzerindeki etkisini araştır ve bulguları analiz et",
        "AB'de yapay zeka düzenlemelerinin mevcut durumu nedir ve startupları nasıl etkiliyor?"
    ]
    
    for query in queries:
        print(f"\nSorgu işleniyor: {query}")
        execute_research_task(query)
```

### Örnek 2: Finansal Veri Analizi

Bu örnek, finansal veri için Alpha Vantage API ve piyasa verisi için Yahoo Finance kullanır:

```python
import ollama
import requests
import pandas as pd
from typing import Dict, Any
import yfinance as yf
from datetime import datetime, timedelta

# API Anahtarları (kendi anahtarlarınızla değiştirin)
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key"  # https://www.alphavantage.co/ adresinden alın

def get_stock_data(symbol: str) -> str:
    """Yahoo Finance kullanarak hisse senedi verisi al"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1mo")
        
        current_price = info.get('currentPrice', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        volume = info.get('volume', 'N/A')
        
        return f"""
{symbol} için Hisse Senedi Verisi:
Mevcut Fiyat: ${current_price}
Piyasa Değeri: ${market_cap:,.2f}
Hacim: {volume:,.0f}
1 Aylık En Yüksek: ${hist['High'].max():.2f}
1 Aylık En Düşük: ${hist['Low'].min():.2f}
"""
    except Exception as e:
        return f"{symbol} için hisse senedi verisi alınırken hata: {str(e)}"

def get_forex_data(from_symbol: str, to_symbol: str) -> str:
    """Alpha Vantage API kullanarak döviz verisi al"""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_symbol,
        "to_currency": to_symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if "Realtime Currency Exchange Rate" in data:
        rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        return f"Mevcut {from_symbol}/{to_symbol} döviz kuru: {rate}"
    else:
        return f"Döviz verisi alınırken hata: {data.get('Note', 'Bilinmeyen hata')}"

def analyze_financial_data(data: str) -> str:
    """Ollama kullanarak finansal veriyi analiz et"""
    response = ollama.chat(
        host="http://your-ollama-url:11434",  # Ollama URL'nizi buraya yazın
        model='llama3.1',
        messages=[{
            'role': 'user',
            'content': f"Bu finansal veriyi analiz et ve önemli içgörüler sağla:\n\n{data}"
        }]
    )
    return response['message']['content']

def execute_financial_analysis(query: str):
    """Birden fazla araç kullanarak bir finansal analiz görevi yürüt"""
    # Araçları tanımla
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_stock_data',
                'description': 'Bir sembol için borsa verisi al',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'symbol': {
                            'type': 'string',
                            'description': 'Hisse senedi sembolü (örn. AAPL, GOOGL)',
                        }
                    },
                    'required': ['symbol'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_forex_data',
                'description': 'Döviz kuru verisi al',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'from_symbol': {
                            'type': 'string',
                            'description': 'Kaynak para birimi sembolü (örn. USD, EUR)',
                        },
                        'to_symbol': {
                            'type': 'string',
                            'description': 'Hedef para birimi sembolü (örn. USD, EUR)',
                        }
                    },
                    'required': ['from_symbol', 'to_symbol'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'analyze_financial_data',
                'description': 'Finansal veriyi analiz et ve içgörüler sağla',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'data': {
                            'type': 'string',
                            'description': 'Analiz edilecek finansal veri',
                        }
                    },
                    'required': ['data'],
                },
            },
        }
    ]

    # İlk konuşma
    conversation = [
        {'role': 'user', 'content': query}
    ]

    # İlk API çağrısı
    response = ollama.chat(
        model='llama3.1',
        messages=conversation,
        tools=tools
    )

    # Araç çağrılarını işle
    if 'tool_calls' in response['message']:
        conversation.append(response['message'])
        
        for tool_call in response['message']['tool_calls']:
            function_name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            
            if function_name == 'get_stock_data':
                stock_data = get_stock_data(args['symbol'])
                conversation.append({'role': 'tool', 'content': stock_data})
            
            elif function_name == 'get_forex_data':
                forex_data = get_forex_data(args['from_symbol'], args['to_symbol'])
                conversation.append({'role': 'tool', 'content': forex_data})
            
            elif function_name == 'analyze_financial_data':
                analysis = analyze_financial_data(args['data'])
                conversation.append({'role': 'tool', 'content': analysis})

        # Son yanıtı al
        final_response = ollama.chat(
            model='llama3.1',
            messages=conversation
        )
        print(final_response['message']['content'])

# Örnek kullanım
if __name__ == "__main__":
    # Alpha Vantage API anahtarınızla değiştirin
    ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key"
    
    # Örnek sorgular
    queries = [
        "AAPL ve GOOGL hisse senetlerinin performansını analiz et ve karşılaştır",
        "Mevcut EUR/USD döviz kuru nedir ve Avrupa ihracatını nasıl etkiliyor?",
        "Son piyasa trendlerinin teknoloji hisseleri üzerindeki etkisini araştır ve içgörüler sağla"
    ]
    
    for query in queries:
        print(f"\nSorgu işleniyor: {query}")
        execute_financial_analysis(query)
```

### Örnek 3: Haber ve Duygu Analizi

Bu örnek, haber verisi için NewsAPI ve duygu analizi için NLTK kullanır:

```python
import ollama
import requests
from typing import Dict, Any
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# API Anahtarları (kendi anahtarlarınızla değiştirin)
NEWS_API_KEY = "your_news_api_key"  # https://newsapi.org/ adresinden alın

# Gerekli NLTK verilerini indir
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_news(query: str) -> str:
    """NewsAPI kullanarak haber makaleleri al"""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "sortBy": "relevancy",
        "language": "en"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code == 200 and data["articles"]:
        articles = data["articles"][:5]  # İlk 5 makaleyi al
        results = []
        for article in articles:
            results.append(f"""
Başlık: {article['title']}
Kaynak: {article['source']['name']}
Yayın Tarihi: {article['publishedAt']}
URL: {article['url']}
""")
        return "\n".join(results)
    else:
        return f"Haber alınırken hata: {data.get('message', 'Bilinmeyen hata')}"

def analyze_sentiment(text: str) -> str:
    """NLTK kullanarak metin duygu analizi yap"""
    try:
        sentiment = sia.polarity_scores(text)
        return f"""
Duygu Analizi Sonuçları:
Pozitif: {sentiment['pos']:.2f}
Negatif: {sentiment['neg']:.2f}
Nötr: {sentiment['neu']:.2f}
Bileşik: {sentiment['compound']:.2f}
"""
    except Exception as e:
        return f"Duygu analizi yapılırken hata: {str(e)}"

def summarize_text(text: str) -> str:
    """Ollama kullanarak metni özetle"""
    response = ollama.chat(
        host="http://your-ollama-url:11434",  # Ollama URL'nizi buraya yazın
        model='llama3.1',
        messages=[{
            'role': 'user',
            'content': f"Bu metni özlü bir şekilde özetle:\n\n{text}"
        }]
    )
    return response['message']['content']

def execute_news_analysis(query: str):
    """Birden fazla araç kullanarak bir haber analizi görevi yürüt"""
    # Araçları tanımla
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_news',
                'description': 'Bir konu hakkında son haber makalelerini al',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'Haber arama sorgusu',
                        }
                    },
                    'required': ['query'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'analyze_sentiment',
                'description': 'Metnin duygu analizini yap',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'Analiz edilecek metin',
                        }
                    },
                    'required': ['text'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'summarize_text',
                'description': 'Metni özetle',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'Özetlenecek metin',
                        }
                    },
                    'required': ['text'],
                },
            },
        }
    ]

    # İlk konuşma
    conversation = [
        {'role': 'user', 'content': query}
    ]

    # İlk API çağrısı
    response = ollama.chat(
        model='llama3.1',
        messages=conversation,
        tools=tools
    )

    # Araç çağrılarını işle
    if 'tool_calls' in response['message']:
        conversation.append(response['message'])
        
        for tool_call in response['message']['tool_calls']:
            function_name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            
            if function_name == 'get_news':
                news_data = get_news(args['query'])
                conversation.append({'role': 'tool', 'content': news_data})
            
            elif function_name == 'analyze_sentiment':
                sentiment = analyze_sentiment(args['text'])
                conversation.append({'role': 'tool', 'content': sentiment})
            
            elif function_name == 'summarize_text':
                summary = summarize_text(args['text'])
                conversation.append({'role': 'tool', 'content': summary})

        # Son yanıtı al
        final_response = ollama.chat(
            model='llama3.1',
            messages=conversation
        )
        print(final_response['message']['content'])

# Örnek kullanım
if __name__ == "__main__":
    # NewsAPI anahtarınızla değiştirin
    NEWS_API_KEY = "your_news_api_key"
    
    # Örnek sorgular
    queries = [
        "Yapay zeka hakkındaki son haberlerin duygu analizini yap",
        "Yenilenebilir enerji hakkındaki son gelişmeleri özetle ve kamuoyu duygu analizini yap",
        "İklim değişikliği hakkında son haberleri araştır ve kapsamlı bir analiz sağla"
    ]
    
    for query in queries:
        print(f"\nSorgu işleniyor: {query}")
        execute_news_analysis(query)
```

Bu örnekleri kullanmak için:

1. Gerekli paketleri yükleyin:
```bash
pip install ollama requests pandas yfinance nltk
```

2. Servisler için API anahtarları alın:
   - OpenWeatherMap: https://openweathermap.org/api
   - Alpha Vantage: https://www.alphavantage.co/
   - NewsAPI: https://newsapi.org/

3. Kod içindeki API anahtar yer tutucularını kendi API anahtarlarınızla değiştirin.

4. Örnekleri çalıştırın:
```bash
python ollama_tool_functions.py
```

Bu örnekler, Ollama'nın araç fonksiyonu sisteminin gerçek API'ler ve servislerle gerçek dünya kullanımını gösterir. Her örnek farklı araç entegrasyon yönlerini gösterir:

1. Web arama örneği, web araması, hava durumu verisi ve metin analizini nasıl birleştireceğinizi gösterir.
2. Finansal analiz örneği, borsa ve döviz verisiyle nasıl çalışacağınızı gösterir.
3. Haber analizi örneği, haber alımını duygu analizi ve özetleme ile nasıl entegre edeceğinizi gösterir.

Her örnek şunları içerir:
- Gerçek API entegrasyonları
- Hata yönetimi
- Veri işleme
- Çoklu araç kullanımı
- Konuşma yönetimi
- Sonuç toplama

Kod çalıştırılmaya hazırdır ve gerektiğinde ek araçlar ve yeteneklerle genişletilebilir.

## Sonuç

Ollama'nın araç fonksiyonu uygulaması şunları birleştiren sofistike bir sistemdir:

1. **Esnek Şablonlar**: Model-spesifik istem biçimlendirmeye olanak sağlar
2. **Akıllı Ayrıştırma**: Model çıktılarından yapılandırılmış veri çıkarır
3. **Standart API**: Geliştiriciler için tutarlı arayüzler sağlar

Bu mimari, yerel LLM'lerin metin üretimi ötesinde harici araçlar ve sistemlerle etkileşime girmesine olanak sağlar. Uygulama şu şekilde tasarlanmıştır:

- **Model-bağımsız**: Araç çağrılarını destekleyen farklı modellerle çalışır
- **Format-esnek**: Çeşitli araç çağrı temsillerine uyum sağlar
- **Geliştirici-dostu**: Kullanışlı API'ler ve OpenAI uyumluluğu sağlar

LLM'ler gelişmeye devam ettikçe, Ollama'nın araç fonksiyonu uygulaması, harici araçlar ve servislerle sorunsuz entegre olabilen daha yetenekli AI sistemleri oluşturmak için sağlam bir temel sağlar. 
