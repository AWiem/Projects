package com.pets.pets_service.Controllers;
import com.pets.pets_service.Exception.ResourceNotFoundException;
import com.pets.pets_service.Models.Product;
import com.pets.pets_service.Repositories.ProductRepo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@CrossOrigin(origins = "http://localhost:4200")
@RestController
@RequestMapping("api/v1")
public class ProductController {
    @Autowired
    private ProductRepo productRepo;

    @GetMapping("/products")
    public List<Product> getAllProducts() {
        return productRepo.findAll();
    }
    @GetMapping("/products/{id}")
    public ResponseEntity<Product> getProductsById(@PathVariable(value = "id") Integer productId)
            throws ResourceNotFoundException {

        Product product = productRepo.findById(productId)
                .orElseThrow(() -> new ResourceNotFoundException("Product not found"));
        return ResponseEntity.ok().body(product);
    }

    @PostMapping("/products")
    public Product createProduct(@Valid @RequestBody Product product) {
        return productRepo.save(product);
    }

    @PutMapping("/products/{id}")
    public ResponseEntity<Product> updateProduct(@PathVariable(value = "id") Integer productId,
                                               @Valid @RequestBody Product productDetails) throws ResourceNotFoundException {
        Product product = productRepo.findById(productId)
                .orElseThrow(() -> new ResourceNotFoundException("Product not found"));
        product.setProductProviders(productDetails.getProductProviders());
        product.setId(productDetails.getId());
        product.setPrice(productDetails.getPrice());
        product.setInStock(productDetails.isInStock());
        product.setProductType(productDetails.getProductType());
        product.setProductProviders(productDetails.getProductProviders());

        final Product updatedProduct = productRepo.save(product);
        return ResponseEntity.ok(updatedProduct);
    }
    @DeleteMapping("/products/{id}")
    public Map<String, Boolean> deleteProduct(@PathVariable(value = "id") Integer productId)
            throws ResourceNotFoundException {
       Product product = productRepo.findById(productId)
                .orElseThrow(() -> new ResourceNotFoundException("Product not found"));
        productRepo.delete(product);
        Map<String, Boolean> response = new HashMap<>();
        response.put("deleted", Boolean.TRUE);
        return response;
    }
}
